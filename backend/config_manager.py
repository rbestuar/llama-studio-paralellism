"""Configuration management for app and models."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from gguf_metadata_parser import parse_gguf_metadata

logger = logging.getLogger(__name__)


KV_BYTES: Dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q6_k": 0.75,
    "q5_k": 0.625,
    "q5_0": 0.625,
    "q4_k": 0.5,
    "q4_0": 0.5,
    "q3_k": 0.375,
}


def calculate_kv_cache_gb(
    block_count: int,
    ctx_size: int,
    kv_cache_multiplier: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
) -> float:
    """Compute KV cache size in GiB.

    Formula: block_count * ctx_size * kv_cache_multiplier * (k_bytes + v_bytes) / 1024^3
    The kv_cache_multiplier is pre-adjusted by the GGUF parser for GQA/MQA/hybrid architectures.

    Uses 1024**3 (GiB) to match file size units from _get_gguf_size_gb and GPU memory
    reported by pynvml, so totals and availability checks are in consistent units.
    """
    k = KV_BYTES.get(cache_type_k, 2.0)
    v = KV_BYTES.get(cache_type_v, 2.0)
    return block_count * ctx_size * kv_cache_multiplier * (k + v) / (1024 ** 3)


def _sanitize_html_id(name: str) -> str:
    """Convert a model name to a safe HTML ID.

    Replaces spaces, periods, and other special characters with underscores
    to ensure the ID is valid for HTML, CSS selectors, and JavaScript.
    """
    # Replace spaces, periods, and other non-alphanumeric chars with underscores
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove leading digits (invalid in CSS ID selectors)
    safe = re.sub(r'^[0-9]+', '', safe)
    # If empty after sanitization, use a fallback
    return safe if safe else 'model'


class ModelConfig(BaseModel):
    """Single model configuration."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Unique model identifier")
    display_name: Optional[str] = Field(default=None)
    html_id: Optional[str] = Field(default=None, description="Safe HTML/CSS identifier (auto-generated from name)")
    model_path: str = Field(..., description="Path to model file (absolute or relative)")
    llama_path: Optional[str] = Field(default="default", description="Custom llama-server binary path, or 'default' to use global config")
    block_count: Optional[int] = Field(default=None, description="Number of transformer layers (from GGUF parser)")
    max_context: Optional[int] = Field(default=None, description="Maximum context length supported by model (from GGUF parser)")
    kv_cache_multiplier: Optional[int] = Field(default=None, description="KV cache dimension multiplier (computed by parser based on attention architecture)")
    launch_args: Optional[Dict[str, Any]] = Field(default=None, description="Launch arguments for llama-server (includes --host, --port)")
    size_gb: Optional[float] = Field(default=None, description="Model size in GB (COMPUTED from file, never persisted to JSON)")
    total_vram: Optional[float] = Field(default=None, description="Total VRAM needed in GB (COMPUTED from file size + KV cache, never persisted to JSON)")
    is_configured: bool = Field(default=False, description="Whether model has full configuration (--port not null in launch_args)")

    def model_post_init(self, __context):
        """Auto-generate html_id from name if not provided."""
        if not self.html_id:
            self.html_id = _sanitize_html_id(self.name)

    @property
    def port(self) -> Optional[int]:
        """Get port from launch_args (computed, not persisted)."""
        if self.launch_args is None:
            return None
        port_val = self.launch_args.get("--port")
        return int(port_val) if port_val is not None else None


class AppConfig(BaseModel):
    """Application configuration."""

    webui_port: int = Field(default=7999, description="WebUI port")
    models_directory: str = Field(default="./models", description="Path to models directory")
    logs_directory: str = Field(default="./logs", description="Path to logs directory")
    llama_server_binary: str = Field(
        default="/home/m6/servers/llamacpp/bin/llama-server",
        description="Path to llama-server binary"
    )
    llama_schema_version: Optional[str] = Field(
        default=None,
        description="Detected llama-server version (e.g., 9030_17df5830e)"
    )

    @field_validator("webui_port")
    def webui_port_must_be_valid(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("WebUI port must be between 1 and 65535")
        return v


class ConfigManager:
    """Manages application and model configurations."""

    def __init__(self, app_config_path: Path = Path("config/app.json"), project_root: Path = None):
        self.app_config_path = Path(app_config_path)
        # If project_root not provided, infer from app_config_path
        if project_root is None:
            self.project_root = self.app_config_path.parent.parent
        else:
            self.project_root = Path(project_root)
        self.app_config: Optional[AppConfig] = None
        self.models: Dict[str, ModelConfig] = {}

    async def load_app_config(self) -> AppConfig:
        """Load application configuration from JSON."""
        try:
            with open(self.app_config_path) as f:
                config_data = json.load(f)
            self.app_config = AppConfig(**config_data)
            logger.info(f"✓ Loaded app config from {self.app_config_path}")
            return self.app_config
        except FileNotFoundError:
            logger.warning(f"⚠ App config not found at {self.app_config_path}, using defaults")
            # Create default config instead of crashing
            self.app_config = AppConfig()
            return self.app_config
        except Exception as e:
            logger.warning(f"⚠ Error loading app config: {e}, using defaults")
            # Use defaults on any error
            self.app_config = AppConfig()
            return self.app_config

    async def save_app_config(self) -> None:
        """Save application configuration to JSON."""
        try:
            with open(self.app_config_path, "w") as f:
                json.dump(self.app_config.model_dump(), f, indent=2)
            logger.info(f"✓ Saved app config to {self.app_config_path}")
        except Exception as e:
            logger.error(f"✗ Error saving app config: {e}")
            raise

    async def load_all_models(self) -> Dict[str, ModelConfig]:
        """
        Load all model configurations.
        
        Process:
        1. Recursively search models_directory for GGUF files
        2. For each GGUF, check for corresponding JSON config in config/models/
        3. If JSON exists with launch_args, load as fully configured model
        4. If JSON exists without launch_args, load as skeleton config
        5. If no JSON exists, generate skeleton config from GGUF metadata
        """
        if not self.app_config:
            raise RuntimeError("App config not loaded yet. Call load_app_config() first.")

        # Resolve models_directory (root of GGUF files)
        models_root = Path(self.app_config.models_directory)
        if not models_root.is_absolute():
            models_root = self.project_root / models_root
        
        # Resolve config directory (always config/models)
        config_models_dir = self.project_root / "config" / "models"
        config_models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        
        # Step 1: Find all GGUF files recursively
        logger.info(f"🔍 Searching for GGUF files in: {models_root}")
        gguf_files = list(models_root.rglob("*.gguf")) if models_root.exists() else []
        logger.info(f"   Found {len(gguf_files)} GGUF file(s)")
        
        for gguf_path in sorted(gguf_files):
            try:
                # Generate model name from filename (without .gguf)
                model_name = gguf_path.stem
                
                # Skip mmproj (multimodal projector) models - they're not standalone chat models
                if model_name.startswith("mmproj"):
                    logger.debug(f"   Skipping mmproj model: {model_name}")
                    continue
                
                config_file = config_models_dir / f"{model_name}.json"
                
                # Step 2: Check for existing JSON config
                if config_file.exists():
                    logger.debug(f"   Loading config for {model_name} from {config_file}")
                    with open(config_file) as f:
                        config_data = json.load(f)

                    config_data = self._backfill_metadata(config_data, gguf_path, config_file)

                    logger.debug(f"      Raw JSON launch_args: {config_data.get('launch_args', {})}")
                    model_config = ModelConfig(**config_data)
                    logger.debug(f"      Loaded launch_args: {model_config.launch_args}")
                    # Mark as configured if --port is not null
                    model_config.is_configured = (
                        model_config.launch_args is not None and
                        model_config.launch_args.get("--port") is not None
                    )
                else:
                    # Step 3: Generate skeleton config from GGUF
                    logger.debug(f"   Generating skeleton config for {model_name}")

                    # Extract GGUF metadata only when creating new skeleton (not on every scan)
                    gguf_meta = parse_gguf_metadata(str(gguf_path))

                    model_config = ModelConfig(
                        name=model_name,
                        display_name=self._format_display_name(model_name),
                        model_path=str(gguf_path),
                        block_count=gguf_meta.block_count,
                        max_context=gguf_meta.max_context,
                        kv_cache_multiplier=gguf_meta.kv_cache_multiplier,
                        launch_args={
                            "--host": None,
                            "--port": None,
                            "--gpu-layers": "999",
                            "--ctx-size": "4096",
                            "--cache-type-k": "f16",
                            "--cache-type-v": "f16",
                            "--batch-size": "512",
                            "--threads": "12",
                        },
                        is_configured=False,
                    )

                    # Save skeleton config
                    self._save_skeleton_config(config_file, model_config, gguf_meta)

                # Compute size_gb from actual file (as double-check that file exists)
                model_config.size_gb = self._get_gguf_size_gb(gguf_path)

                # Calculate total VRAM needed (file + KV cache + overhead)
                model_config.total_vram = self._calculate_total_vram(model_config)

                self.models[model_name] = model_config
                status = "✓ configured" if model_config.is_configured else "⚠ skeleton"
                logger.info(f"   {status}: {model_name} ({model_config.size_gb:.1f}GB)")
                
            except Exception as e:
                logger.error(f"✗ Error processing {gguf_path}: {e}")

        logger.info(f"✓ Loaded {len(self.models)} model(s) ({sum(1 for m in self.models.values() if m.is_configured)} configured, {sum(1 for m in self.models.values() if not m.is_configured)} skeleton)")
        return self.models

    def rescan_model(self, model_name: str) -> Optional[ModelConfig]:
        """Rescan a specific model: reload config from JSON, re-extract GGUF metadata, recompute size_gb, re-evaluate is_configured."""
        try:
            models_root = self.project_root / self.app_config.models_directory
            config_models_dir = self.project_root / "config" / "models"

            # Find GGUF file for this model
            gguf_files = list(models_root.rglob(f"{model_name}.gguf"))
            if not gguf_files:
                logger.warning(f"⚠ Model file not found: {model_name}.gguf")
                return None

            gguf_path = gguf_files[0]
            config_file = config_models_dir / f"{model_name}.json"

            # Extract fresh GGUF metadata
            gguf_meta = parse_gguf_metadata(str(gguf_path))

            # Load or create config (same logic as load_all_models)
            if config_file.exists():
                logger.debug(f"   Reloading config from {config_file}")
                with open(config_file) as f:
                    config_data = json.load(f)

                config_data = self._backfill_metadata(config_data, gguf_path, config_file, gguf_meta)

                model_config = ModelConfig(**config_data)
                model_config.is_configured = (
                    model_config.launch_args is not None and
                    model_config.launch_args.get("--port") is not None
                )
            else:
                logger.warning(f"⚠ No config found for {model_name}, creating skeleton")
                model_config = ModelConfig(
                    name=model_name,
                    display_name=self._format_display_name(model_name),
                    model_path=str(gguf_path),
                    block_count=gguf_meta.block_count,
                    max_context=gguf_meta.max_context,
                    kv_cache_multiplier=gguf_meta.kv_cache_multiplier,
                    launch_args={
                        "--host": None,
                        "--port": None,
                        "--gpu-layers": "999",
                        "--ctx-size": "4096",
                        "--cache-type-k": "f16",
                        "--cache-type-v": "f16",
                        "--batch-size": "512",
                        "--threads": "12",
                    },
                    is_configured=False,
                )
                self._save_skeleton_config(config_file, model_config, gguf_meta)

            # Compute size_gb from actual file
            model_config.size_gb = self._get_gguf_size_gb(gguf_path)

            # Calculate total VRAM needed (file + KV cache + overhead)
            model_config.total_vram = self._calculate_total_vram(model_config)

            # Update in-memory cache
            self.models[model_name] = model_config
            status = "✓ configured" if model_config.is_configured else "⚠ skeleton"
            logger.info(f"   Rescanned {status}: {model_name} ({model_config.size_gb:.1f}GB)")

            return model_config

        except Exception as e:
            logger.error(f"✗ Error rescanning {model_name}: {e}")
            return None

    def _backfill_metadata(
        self,
        config_data: dict,
        gguf_path: Path,
        config_file: Path,
        gguf_meta=None,
    ) -> dict:
        """Ensure GGUF metadata fields are present in config_data.

        If any of block_count/max_context/kv_cache_multiplier are missing, extract
        them from the GGUF file, insert them in the canonical field order, and save
        the updated JSON back to disk.

        If all fields are already present and gguf_meta was pre-parsed by the caller
        (e.g. rescan_model), merge the fresh values into config_data in-memory so the
        returned dict is always up-to-date.

        The gguf_meta argument is optional: pass it when already parsed (avoids a
        redundant subprocess call), or leave it None to parse lazily only when needed.
        """
        needs_metadata = not all(
            config_data.get(k) is not None
            for k in ("block_count", "max_context", "kv_cache_multiplier")
        )

        if needs_metadata:
            if gguf_meta is None:
                gguf_meta = parse_gguf_metadata(str(gguf_path))
            meta_dict = vars(gguf_meta)
            # Rebuild dict with metadata inserted after model_path
            reordered: dict = {}
            for key in ("name", "display_name", "model_path"):
                if key in config_data:
                    reordered[key] = config_data[key]
            for key in ("block_count", "max_context", "kv_cache_multiplier"):
                reordered[key] = meta_dict.get(key)
            for key, value in config_data.items():
                if key not in reordered:
                    reordered[key] = value
            config_data = reordered
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.debug(f"   Saved GGUF metadata to {config_file}")
        elif gguf_meta is not None:
            # Caller already parsed fresh metadata — update in-memory (not persisted)
            config_data.update(vars(gguf_meta))

        return config_data

    def _get_gguf_size_gb(self, gguf_path: Path) -> float:
        """Get GGUF file size in GB."""
        try:
            size_bytes = gguf_path.stat().st_size
            size_gb = size_bytes / (1024**3)
            return round(size_gb, 1)
        except Exception as e:
            logger.warning(f"⚠ Could not get size for {gguf_path}: {e}")
            return 0.0

    def _calculate_total_vram(self, model_config: ModelConfig) -> Optional[float]:
        """Calculate total VRAM needed: weights + KV cache.

        Returns None if insufficient data (missing size_gb, block_count, kv_cache_multiplier).
        """
        if not model_config.size_gb or not model_config.block_count or not model_config.kv_cache_multiplier:
            return None
        if not model_config.launch_args:
            return None

        ctx_size = model_config.launch_args.get("--ctx-size") or model_config.launch_args.get("-c") or 4096
        cache_type_k = model_config.launch_args.get("--cache-type-k") or model_config.launch_args.get("-ctk") or "f16"
        cache_type_v = model_config.launch_args.get("--cache-type-v") or model_config.launch_args.get("-ctv") or "f16"

        try:
            ctx_size = int(ctx_size)
        except (ValueError, TypeError):
            ctx_size = 4096

        kv_cache_gb = calculate_kv_cache_gb(
            model_config.block_count, ctx_size, model_config.kv_cache_multiplier,
            str(cache_type_k), str(cache_type_v),
        )
        return round(model_config.size_gb + kv_cache_gb, 2)

    def _format_display_name(self, filename: str) -> str:
        """Format filename into a readable display name."""
        # Remove .gguf extension and common quantization suffixes
        name = filename.replace('.gguf', '')
        # Replace hyphens and underscores with spaces
        name = name.replace('-', ' ').replace('_', ' ')
        # Title case
        return ' '.join(word.capitalize() for word in name.split())

    def _save_skeleton_config(self, config_file: Path, model_config: ModelConfig, gguf_meta: dict = None) -> None:
        """Save skeleton configuration with minimal GGUF metadata (block_count, max_context, kv_cache_multiplier)."""
        try:
            if gguf_meta is None:
                gguf_meta = {}
            gguf_meta_dict = vars(gguf_meta) if gguf_meta else {}
            config_data = {
                "name": model_config.name,
                "display_name": model_config.display_name,
                "model_path": model_config.model_path,
                "block_count": gguf_meta_dict.get("block_count"),
                "max_context": gguf_meta_dict.get("max_context"),
                "kv_cache_multiplier": gguf_meta_dict.get("kv_cache_multiplier"),
                "launch_args": {
                    "--host": None,
                    "--port": None,
                    "--gpu-layers": "999",
                    "--ctx-size": "4096",
                    "--cache-type-k": "f16",
                    "--cache-type-v": "f16",
                    "--batch-size": "512",
                    "--threads": "12",
                },
            }
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.debug(f"   Saved skeleton config: {config_file}")
        except Exception as e:
            logger.error(f"✗ Error saving skeleton config {config_file}: {e}")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self.models:
            raise FileNotFoundError(f"Model config not found: {model_name}")
        return self.models[model_name]

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all loaded model configurations."""
        return self.models

    def get_log_path(self, model_name: str) -> Path:
        """Get log file path for a model."""
        if not self.app_config:
            raise RuntimeError("App config not loaded yet.")

        logs_dir = Path(self.app_config.logs_directory)
        if not logs_dir.is_absolute():
            logs_dir = self.project_root / logs_dir

        return logs_dir / f"{model_name}.log"

    def needs_configuration(self) -> tuple[bool, Optional[str]]:
        """
        Check if configuration is incomplete (first install or corrupted).

        Returns: (needs_config, error_message)
        - True if llama_server_binary doesn't exist or models_directory is empty
        - Returns reason for first-time UI
        """
        if not self.app_config:
            return True, "Configuration not loaded"

        if not self.app_config.llama_server_binary:
            return True, "llama-server binary not configured"

        binary_path = Path(self.app_config.llama_server_binary)
        if not binary_path.exists():
            return True, f"llama-server binary not found: {self.app_config.llama_server_binary}"

        models_path = Path(self.app_config.models_directory)
        if not models_path.is_absolute():
            models_path = self.project_root / models_path

        if not models_path.exists():
            return True, f"Models directory not found: {self.app_config.models_directory}"

        return False, None

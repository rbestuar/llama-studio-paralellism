"""GGUF metadata parser for model architecture detection and KV cache multiplier computation.

This module handles the complexity of different model architectures:
- Standard multi-head attention
- GQA (Grouped Query Attention) and MQA (Multi-Query Attention)
- Hybrid attention+SSM models (e.g., Qwen3.6)

The parser extracts metadata once and returns a pre-adjusted kv_cache_multiplier
that works correctly with the standard VRAM formula, accounting for all architecture nuances.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GGUFMetadata:
    """Parsed GGUF metadata for VRAM calculation.

    All values are pre-adjusted for the model's architecture, so the VRAM formula
    can remain simple and uniform across all models.
    """
    block_count: Optional[int] = None
    max_context: Optional[int] = None
    kv_cache_multiplier: Optional[int] = None


def parse_gguf_metadata(model_path: str) -> GGUFMetadata:
    """Extract and compute GGUF metadata for VRAM calculation.

    Handles:
    - Standard attention: multiplier = hidden_dim
    - GQA/MQA: multiplier = num_heads_kv * head_dim
    - Hybrid attention+SSM: multiplier = (num_heads_kv * head_dim) * (kv_active_layers / total_layers)

    Returns GGUFMetadata with pre-adjusted multiplier that works with standard formula:
        kv_cache = block_count * ctx_size * kv_cache_multiplier * (k_bytes + v_bytes) / 1e9
    """
    try:
        import time
        start_total = time.time()

        from gguf_parser import GGUFParser

        logger.debug(f"   [GGUF] Parsing {model_path}...")
        parser = GGUFParser(model_path)
        parser.parse()
        parse_time = time.time() - start_total
        logger.debug(f"   [GGUF] Parse complete in {parse_time*1000:.1f}ms, found {len(parser.metadata)} fields")

        metadata = parser.metadata

        # Detect architecture to use correct field name prefixes
        architecture = metadata.get("general.architecture", "llama")
        arch_prefix = f"{architecture}."

        logger.debug(f"   [GGUF] Architecture: {architecture}")

        # Extract required fields
        hidden_dim = (
            metadata.get(f"{arch_prefix}embedding_length") or
            metadata.get("llama.embedding_length")
        )

        block_count = (
            metadata.get(f"{arch_prefix}block_count") or
            metadata.get("llama.block_count")
        )

        # Extract max context length (supported by model during training)
        max_context = (
            metadata.get(f"{arch_prefix}context_length") or
            metadata.get("llama.context_length")
        )

        num_heads = (
            metadata.get(f"{arch_prefix}attention.head_count") or
            metadata.get("llama.attention.head_count")
        )

        num_heads_kv = (
            metadata.get(f"{arch_prefix}attention.head_count_kv") or
            metadata.get("llama.attention.head_count_kv")
        )

        # Check for direct KV embedding size (some models like Qwen3.6 have explicit values)
        n_embd_k_gqa = (
            metadata.get(f"{arch_prefix}embedding.length_kv") or
            metadata.get("llama.embedding.length_kv") or
            metadata.get("n_embd_k_gqa")
        )

        # Detect hybrid attention+SSM models (e.g., Qwen3.6)
        # These have ssm_group_count indicating number of attention layers
        ssm_group_count = metadata.get(f"{arch_prefix}ssm.group_count")
        kv_active_layers = block_count

        if ssm_group_count and block_count and ssm_group_count < block_count:
            # Hybrid model: only some layers use attention+KV
            # ssm_group_count indicates the number of attention/KV layers
            # (remaining layers use SSM/Mamba, which don't use KV cache)
            kv_active_layers = ssm_group_count
            logger.debug(f"   [GGUF] Detected hybrid attention+SSM: block_count={block_count}, attention_layers={kv_active_layers}, ssm_group_count={ssm_group_count}")

        if block_count:
            logger.debug(f"   [GGUF] block_count={block_count}, kv_active_layers={kv_active_layers}")
        if max_context:
            logger.debug(f"   [GGUF] max_context={max_context}")

        # Compute KV cache multiplier based on attention architecture
        kv_cache_multiplier = None

        # If we have the actual KV embedding size, use it directly
        if n_embd_k_gqa:
            kv_cache_multiplier = n_embd_k_gqa
            logger.debug(f"   [GGUF] Using direct KV embedding: n_embd_k_gqa={n_embd_k_gqa}")
        elif hidden_dim and num_heads:
            # Handle per-layer arrays (e.g., Gemma-4, Deci models): use first element
            if isinstance(num_heads, list):
                num_heads = num_heads[0] if num_heads else None
            if isinstance(num_heads_kv, list):
                num_heads_kv = num_heads_kv[0] if num_heads_kv else None

            if num_heads and num_heads_kv and num_heads_kv < num_heads:
                # GQA or MQA detected
                head_dim = hidden_dim // num_heads
                kv_cache_multiplier = num_heads_kv * head_dim
                logger.debug(f"   [GGUF] Detected GQA: num_heads={num_heads}, num_heads_kv={num_heads_kv}, head_dim={head_dim}, multiplier={kv_cache_multiplier}")
            elif num_heads:
                # Standard multi-head attention
                kv_cache_multiplier = hidden_dim
                logger.debug(f"   [GGUF] Detected standard attention: num_heads={num_heads}, multiplier={kv_cache_multiplier}")

        # Pre-adjust multiplier for hybrid architectures
        if kv_cache_multiplier and block_count and kv_active_layers < block_count:
            # Hybrid model: adjust multiplier so formula uses total block_count
            # Formula: block_count * ctx * multiplier * bytes
            # Needs to account for: kv_active_layers * ctx * base_multiplier * bytes
            # So: multiplier = base_multiplier * (kv_active_layers / block_count)
            adjustment_ratio = kv_active_layers / block_count
            kv_cache_multiplier = int(kv_cache_multiplier * adjustment_ratio)
            logger.debug(f"   [GGUF] Adjusted multiplier for hybrid: {kv_cache_multiplier} (ratio={adjustment_ratio:.2f})")

        if kv_cache_multiplier:
            logger.debug(f"   [GGUF] ✓ Extraction complete (block_count={block_count}, max_context={max_context}, kv_cache_multiplier={kv_cache_multiplier})")
        else:
            logger.debug(f"   [GGUF] ⚠ Could not compute kv_cache_multiplier (hidden_dim={hidden_dim}, num_heads={num_heads})")

        return GGUFMetadata(
            block_count=block_count,
            max_context=max_context,
            kv_cache_multiplier=kv_cache_multiplier,
        )
    except Exception as e:
        logger.warning(f"⚠ GGUF metadata extraction failed for {model_path}: {type(e).__name__}: {e}")
        return GGUFMetadata()

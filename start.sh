#!/bin/bash

# Llama Studio Startup Script
# This script activates the Python virtual environment and starts the application

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🦙 Starting Llama Studio..."
echo ""

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ Virtual environment not found at $SCRIPT_DIR/venv"
    echo ""
    echo "To set up the environment, run:"
    echo "  python3 -m venv $SCRIPT_DIR/venv"
    echo "  source $SCRIPT_DIR/venv/bin/activate"
    echo "  pip install -r $SCRIPT_DIR/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Change to backend directory
cd "$SCRIPT_DIR/backend"

# Get configuration
if [ -f "$SCRIPT_DIR/config/app.json" ]; then
    PORT=$(grep -o '"webui_port"[[:space:]]*:[[:space:]]*[0-9]*' "$SCRIPT_DIR/config/app.json" | grep -o '[0-9]*')
    echo "⚙️  WebUI Port: ${PORT:-7999}"
else
    PORT=7999
    echo "⚠️  Config file not found, using default port 7999"
fi

echo ""
echo "🚀 Starting FastAPI server..."
echo "   URL: http://localhost:${PORT:-7999}"
echo "   Press Ctrl+C to stop"
echo ""

# Check for verbose flag
VERBOSE_FLAG=""
if [[ "$1" == "-v" ]] || [[ "$1" == "--verbose" ]]; then
    VERBOSE_FLAG="--verbose"
    echo "🔍 Verbose logging enabled"
    echo ""
fi

# Start uvicorn (pass VERBOSE_FLAG as environment variable, not as argument)
VERBOSE=$VERBOSE_FLAG python -m uvicorn main:app --reload --host 0.0.0.0 --port "${PORT:-7999}"

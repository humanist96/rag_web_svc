#!/bin/bash
# Render startup script - explicitly use Python

echo "Starting AI Nexus Backend..."
echo "Python version:"
python --version
echo "Current directory:"
pwd
echo "Files in directory:"
ls -la

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the application
echo "Starting Uvicorn server..."
python -m uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port ${PORT:-8001}
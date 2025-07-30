#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # exit on error

echo "Starting Python build process..."

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements_deploy.txt

echo "Build completed successfully!"
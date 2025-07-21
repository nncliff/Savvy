#!/bin/bash

# Start script for Breakthrough Discovery API

echo "Starting Breakthrough Discovery API..."
echo "Initializing directories..."

# Create necessary directories
mkdir -p data results

echo "Starting FastAPI server..."
python breakthrough_api.py
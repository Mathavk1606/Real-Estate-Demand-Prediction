#!/bin/bash

set -e

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo ""
    echo "1. Creating virtual environment..."
    python3 -m venv venv
fi

echo ""
echo "2. Activating virtual environment..."
source venv/bin/activate

echo ""
echo "3. Installing dependencies..."
pip install -r requirements.txt --quiet

if [ ! -f "database/real_estate.db" ]; then
    python src/db_setup.py
else
    echo "   Database already exists, skipping..."
fi

if [ ! -f "models/default_scorer.pkl" ]; then
    python models/train_model.py
else
    echo "   Model already exists, skipping..."
fi
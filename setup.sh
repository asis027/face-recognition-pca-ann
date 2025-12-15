#!/bin/bash

# Face Recognition System Setup Script
# This script sets up the environment and prepares the project

echo "=================================="
echo "Face Recognition System Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p dataset
mkdir -p model_output
mkdir -p evaluation_results
mkdir -p visualizations

# Download sample dataset (optional)
echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Place your dataset in the 'dataset' folder"
echo "   Structure: dataset/person_name/image1.jpg"
echo ""
echo "2. Train the model:"
echo "   python train.py --data_dir ./dataset"
echo ""
echo "3. Make predictions:"
echo "   python predict.py --model model_output/model.pkl --image test.jpg"
echo ""
echo "4. Evaluate the model:"
echo "   python evaluate.py --model model_output/model.pkl --data_dir ./dataset --plots"
echo ""
echo "5. Visualize results:"
echo "   python visualize.py --model model_output/model.pkl --all --save"
echo ""
echo "For detailed documentation, see README.md"
echo "=================================="

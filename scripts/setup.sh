#!/bin/bash
# Setup script for Orange Pi RV 2

echo "================================"
echo "Setting up Human Face Detection System"
echo "Optimized for Orange Pi RV 2"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (recommended)
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Setup completed successfully!"
    echo "================================"
    echo ""
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the program:"
    echo "  python main.py --input input/your_file.jpg --pipeline sequential"
    echo ""
    echo "For more options:"
    echo "  python main.py --help"
    echo ""
else
    echo ""
    echo "Error: Installation failed. Please check the error messages above."
    exit 1
fi

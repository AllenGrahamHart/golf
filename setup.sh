#!/bin/bash
set -e

echo "=== Golf Swing Analyzer Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing 4DHumans..."
pip install git+https://github.com/shubham-goel/4D-Humans.git

echo "Installing additional dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download SMPL model from https://smpl.is.tue.mpg.de/"
echo "   Place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in data/smpl/"
echo "3. Run inference: python -m src.inference --image input/your_image.jpg"

#!/bin/bash

# Complete Repository Fix Script for face-recognition-pca-ann
# This script fixes all naming and structure issues

set -e  # Exit on any error

echo "============================================"
echo "  Face Recognition PCA-ANN Repository Fix"
echo "============================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Fixing Python file names...${NC}"

# Rename all Python files
if [ -f "pca_ann.py - Core PCA Module" ]; then
    mv "pca_ann.py - Core PCA Module" "pca_ann.py"
    echo -e "${GREEN}✓ Renamed pca_ann.py${NC}"
fi

if [ -f "train.py - Training Script" ]; then
    mv "train.py - Training Script" "train.py"
    echo -e "${GREEN}✓ Renamed train.py${NC}"
fi

if [ -f "evaluate.py - Model Evaluation" ]; then
    mv "evaluate.py - Model Evaluation" "evaluate.py"
    echo -e "${GREEN}✓ Renamed evaluate.py${NC}"
fi

if [ -f "predict.py - Prediction Script" ]; then
    mv "predict.py - Prediction Script" "predict.py"
    echo -e "${GREEN}✓ Renamed predict.py${NC}"
fi

if [ -f "test_pca_ann.py - Unit Tests" ]; then
    mv "test_pca_ann.py - Unit Tests" "test_pca_ann.py"
    echo -e "${GREEN}✓ Renamed test_pca_ann.py${NC}"
fi

if [ -f "visualize.py - Visualization Tools" ]; then
    mv "visualize.py - Visualization Tools" "visualize.py"
    echo -e "${GREEN}✓ Renamed visualize.py${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Fixing configuration files...${NC}"

if [ -f "config.yaml - Configuration File" ]; then
    mv "config.yaml - Configuration File" "config.yaml"
    echo -e "${GREEN}✓ Renamed config.yaml${NC}"
fi

if [ -f "requirements.txt - Updated" ]; then
    mv "requirements.txt - Updated" "requirements.txt"
    echo -e "${GREEN}✓ Renamed requirements.txt${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Fixing shell scripts...${NC}"

if [ -f "setup.sh - Easy Setup Script" ]; then
    mv "setup.sh - Easy Setup Script" "setup.sh"
    chmod +x setup.sh
    echo -e "${GREEN}✓ Renamed and made setup.sh executable${NC}"
fi

if [ -f "init_repo.sh - Repository Intialazation" ]; then
    mv "init_repo.sh - Repository Intialazation" "init_repo.sh"
    chmod +x init_repo.sh
    echo -e "${GREEN}✓ Renamed and made init_repo.sh executable${NC}"
fi

echo ""
echo -e "${YELLOW}Step 4: Organizing documentation files...${NC}"

# Create docs directory
mkdir -p docs

if [ -f "README.md - Github Optimized" ]; then
    mv "README.md - Github Optimized" "docs/README_github_optimized.md"
    echo -e "${GREEN}✓ Moved README.md - Github Optimized to docs/${NC}"
fi

if [ -f "README.md Complete Document" ]; then
    mv "README.md Complete Document" "docs/README_complete.md"
    echo -e "${GREEN}✓ Moved README.md Complete Document to docs/${NC}"
fi

if [ -f "Github Repository Setup Guide" ]; then
    mv "Github Repository Setup Guide" "docs/github_setup_guide.md"
    echo -e "${GREEN}✓ Moved Github Repository Setup Guide to docs/${NC}"
fi

if [ -f "GITHUB_SETUP_CHEAKLIST.md" ]; then
    mv "GITHUB_SETUP_CHEAKLIST.md" "docs/github_setup_checklist.md"
    echo -e "${GREEN}✓ Fixed CHEAKLIST typo and moved to docs/${NC}"
fi

echo ""
echo -e "${YELLOW}Step 5: Fixing .gitignore files...${NC}"

if [ -f ".gitignore - Git ignore Rules" ]; then
    # Merge the two .gitignore files
    cat ".gitignore - Git ignore Rules" >> .gitignore
    rm ".gitignore - Git ignore Rules"
    echo -e "${GREEN}✓ Merged .gitignore files${NC}"
fi

echo ""
echo -e "${YELLOW}Step 6: Creating proper directory structure...${NC}"

# Create necessary directories
mkdir -p data/{raw,processed,train,test,validation}
mkdir -p models/{saved,checkpoints}
mkdir -p output/{visualizations,reports,logs}
mkdir -p src
mkdir -p tests
mkdir -p notebooks
mkdir -p configs

# Create .gitkeep files to preserve empty directories
touch data/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/train/.gitkeep
touch data/test/.gitkeep
touch data/validation/.gitkeep
touch models/.gitkeep
touch models/saved/.gitkeep
touch models/checkpoints/.gitkeep
touch output/.gitkeep
touch output/visualizations/.gitkeep
touch output/reports/.gitkeep
touch output/logs/.gitkeep
touch tests/.gitkeep
touch notebooks/.gitkeep

echo -e "${GREEN}✓ Created directory structure${NC}"

echo ""
echo -e "${YELLOW}Step 7: Creating/updating .gitignore...${NC}"

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.csv
*.jpg
*.jpeg
*.png
*.pgm
*.gif
*.bmp
*.tiff
*.npy
*.npz

# Model files
models/saved/*
!models/saved/.gitkeep
models/checkpoints/*
!models/checkpoints/.gitkeep
*.h5
*.pkl
*.pickle
*.pth
*.onnx
*.pb

# Output files
output/visualizations/*
!output/visualizations/.gitkeep
output/reports/*
!output/reports/.gitkeep
output/logs/*
!output/logs/.gitkeep
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment variables
.env
.env.local

# OS
Thumbs.db
.DS_Store
EOF

echo -e "${GREEN}✓ Created comprehensive .gitignore${NC}"

echo ""
echo -e "${YELLOW}Step 8: Creating package structure files...${NC}"

# Create __init__.py files
touch src/__init__.py
touch tests/__init__.py

echo -e "${GREEN}✓ Created __init__.py files${NC}"

echo ""
echo -e "${YELLOW}Step 9: Creating main.py entry point...${NC}"

# Create main.py if it doesn't exist
if [ ! -f "main.py" ]; then
    cat > main.py << 'EOF'
"""
Face Recognition System using PCA and ANN
Main entry point for training, evaluation, and prediction
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Face Recognition System using PCA and ANN'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'predict', 'visualize'],
        required=True,
        help='Operation mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file or directory path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory path'
    )
    
    args = parser.parse_args()
    
    # Import modules based on mode
    try:
        if args.mode == 'train':
            from train import train_model
            train_model(args.config)
        elif args.mode == 'evaluate':
            from evaluate import evaluate_model
            evaluate_model(args.config)
        elif args.mode == 'predict':
            from predict import predict
            predict(args.config, args.input)
        elif args.mode == 'visualize':
            from visualize import visualize_results
            visualize_results(args.config)
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all required Python files are present and properly named.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF
    echo -e "${GREEN}✓ Created main.py${NC}"
else
    echo -e "${YELLOW}⚠ main.py already exists, skipping${NC}"
fi

echo ""
echo -e "${YELLOW}Step 10: Verifying file structure...${NC}"

# Check if key files exist
files_to_check=("pca_ann.py" "train.py" "evaluate.py" "predict.py" "config.yaml" "requirements.txt" "README.md")
all_files_present=true

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ Found $file${NC}"
    else
        echo -e "${RED}✗ Missing $file${NC}"
        all_files_present=false
    fi
done

echo ""
echo "============================================"
if [ "$all_files_present" = true ]; then
    echo -e "${GREEN}✓ Repository structure fixed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Some files are still missing${NC}"
fi
echo "============================================"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review the changes: git status"
echo "2. Test imports: python -c 'import pca_ann'"
echo "3. Stage changes: git add ."
echo "4. Commit: git commit -m 'Fix: Rename files and reorganize structure'"
echo "5. Push: git push origin main"
echo ""
echo -e "${GREEN}Done!${NC}"

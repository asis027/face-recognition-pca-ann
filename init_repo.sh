#!/bin/bash

# Repository Initialization Script
# This script sets up the local repository structure and prepares it for GitHub

echo "========================================"
echo "Face Recognition Repository Setup"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    echo "Visit: https://git-scm.com/downloads"
    exit 1
fi

echo "✓ Git is installed"

# Get repository name
read -p "Enter repository name [face-recognition-pca-ann]: " REPO_NAME
REPO_NAME=${REPO_NAME:-face-recognition-pca-ann}

# Check if directory already exists
if [ -d "$REPO_NAME" ]; then
    echo "Error: Directory '$REPO_NAME' already exists!"
    read -p "Remove it and continue? (y/n): " REMOVE
    if [ "$REMOVE" = "y" ] || [ "$REMOVE" = "Y" ]; then
        rm -rf "$REPO_NAME"
        echo "✓ Removed existing directory"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create project directory
mkdir -p "$REPO_NAME"
cd "$REPO_NAME"
echo "✓ Created directory: $REPO_NAME"

# Initialize git repository
git init
echo "✓ Initialized git repository"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p dataset
mkdir -p model_output
mkdir -p evaluation_results
mkdir -p visualizations

# Create .gitkeep files to track empty directories
touch dataset/.gitkeep
touch model_output/.gitkeep
touch evaluation_results/.gitkeep
touch visualizations/.gitkeep

echo "✓ Created directories: dataset, model_output, evaluation_results, visualizations"

# Create __init__.py
touch __init__.py
echo "✓ Created __init__.py"

# Create placeholder README
cat > README.md << 'EOF'
# Face Recognition using PCA + ANN

Production-ready face recognition system. Full documentation coming soon.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --data_dir ./dataset
```

See QUICKSTART.md for detailed instructions.
EOF
echo "✓ Created placeholder README.md"

# Create basic .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
venv/
env/

# Data
dataset/
!dataset/.gitkeep
model_output/
!model_output/.gitkeep
evaluation_results/
!evaluation_results/.gitkeep
visualizations/
!visualizations/.gitkeep
*.jpg
*.png
*.pkl

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF
echo "✓ Created .gitignore"

# Create basic requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.21.0
scipy>=1.7.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
PyYAML>=5.4.0
EOF
echo "✓ Created requirements.txt"

# Stage initial files
git add .gitkeep README.md .gitignore requirements.txt __init__.py
git add dataset/.gitkeep model_output/.gitkeep evaluation_results/.gitkeep visualizations/.gitkeep

echo ""
echo "========================================"
echo "✓ Repository structure created!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy all Python files from the chat into this directory:"
echo "   - pca_ann.py"
echo "   - train.py"
echo "   - predict.py"
echo "   - evaluate.py"
echo "   - visualize.py"
echo "   - test_pca_ann.py"
echo "   - config.yaml"
echo ""
echo "2. Update README.md with the full version from the chat"
echo ""
echo "3. Add and commit files:"
echo "   git add ."
echo "   git commit -m 'Initial commit: Complete face recognition system'"
echo ""
echo "4. Create GitHub repository at: https://github.com/new"
echo "   Name: $REPO_NAME"
echo "   Description: Production-ready face recognition using PCA + ANN"
echo ""
echo "5. Connect to GitHub and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Current directory: $(pwd)"
echo "========================================"

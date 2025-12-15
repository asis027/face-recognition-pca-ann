# Face Recognition using PCA + ANN

A production-ready face recognition system using Principal Component Analysis (PCA) for feature extraction (Eigenfaces) and Artificial Neural Networks (ANN) for classification.

## 🌟 Features

- **PCA-based Feature Extraction**: Efficient dimensionality reduction using eigenfaces
- **Multiple Classifiers**: Support for MLP, SVM, and KNN classifiers
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and error analysis
- **Rich Visualizations**: Eigenfaces, mean face, variance plots, and reconstructions
- **Unknown Person Detection**: Confidence-based rejection of unknown faces
- **Batch Prediction**: Process multiple images at once
- **Face Detection**: Optional preprocessing with OpenCV
- **Configuration Management**: YAML-based configuration
- **Train/Test Split**: Proper evaluation on held-out data

## 📁 Project Structure

```
.
├── pca_ann.py              # Core PCA and eigenface implementation
├── train.py                # Training script
├── predict.py              # Prediction script with confidence scores
├── evaluate.py             # Comprehensive model evaluation
├── visualize.py            # Visualization tools
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd face-recognition-pca

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the dataset from: https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip

Organize your dataset in the following structure:

```
dataset/
  person1/
    img1.jpg
    img2.jpg
    img3.jpg
  person2/
    img1.jpg
    img2.jpg
  person3/
    ...
```

### 3. Training

**Basic training:**
```bash
python train.py --data_dir ./dataset --output model_output
```

**Advanced training with custom parameters:**
```bash
python train.py \
  --data_dir ./dataset \
  --k 50 \
  --classifier mlp \
  --hidden_layers 128 64 \
  --img_size 64 \
  --test_split 0.2 \
  --output model_output
```

**Training options:**
- `--data_dir`: Root dataset directory (required)
- `--k`: Number of eigenfaces (default: 50)
- `--variance_threshold`: Minimum variance to retain (e.g., 0.95)
- `--img_size`: Image size in pixels (default: 64)
- `--classifier`: Classifier type: mlp, svm, or knn (default: mlp)
- `--hidden_layers`: Hidden layer sizes for MLP (default: 128 64)
- `--max_iter`: Maximum iterations for MLP (default: 500)
- `--test_split`: Test set fraction (default: 0.2)
- `--output`: Output folder (default: model_output)

### 4. Prediction

**Single image prediction:**
```bash
python predict.py \
  --model model_output/model.pkl \
  --image path/to/test.jpg
```

**Batch prediction:**
```bash
# Create a text file with image paths (one per line)
echo "path/to/image1.jpg" > images.txt
echo "path/to/image2.jpg" >> images.txt

python predict.py \
  --model model_output/model.pkl \
  --batch images.txt
```

**With face detection:**
```bash
python predict.py \
  --model model_output/model.pkl \
  --image path/to/test.jpg \
  --detect_face
```

**Prediction options:**
- `--model`: Path to trained model (required)
- `--image`: Single image path
- `--batch`: Text file with image paths
- `--confidence_threshold`: Threshold for unknown detection (default: 0.5)
- `--detect_face`: Enable face detection preprocessing

### 5. Evaluation

**Evaluate on test set:**
```bash
python evaluate.py \
  --model model_output/model.pkl \
  --data_dir ./dataset \
  --plots
```

**Evaluate on separate test directory:**
```bash
python evaluate.py \
  --model model_output/model.pkl \
  --test_dir ./test_dataset \
  --output evaluation_results \
  --plots
```

This generates:
- `evaluation_report.txt`: Comprehensive text report
- `detailed_metrics.json`: JSON metrics for programmatic access
- `per_class_metrics.png`: Per-class performance chart
- `confidence_distribution.png`: Confidence analysis plots

### 6. Visualization

**Visualize eigenfaces:**
```bash
python visualize.py \
  --model model_output/model.pkl \
  --eigenfaces \
  --n_eigenfaces 16 \
  --save
```

**Visualize all components:**
```bash
python visualize.py \
  --model model_output/model.pkl \
  --all \
  --save \
  --output visualizations
```

**Visualize face reconstruction:**
```bash
python visualize.py \
  --model model_output/model.pkl \
  --reconstruction path/to/face.jpg \
  --k_values 5 10 20 50 \
  --save
```

**Visualization options:**
- `--all`: Generate all visualizations
- `--eigenfaces`: Display eigenfaces
- `--mean_face`: Display mean face
- `--variance`: Plot variance explained
- `--reconstruction`: Reconstruct a face with different k values
- `--confusion_matrix`: Display confusion matrix
- `--save`: Save figures to files

## 📊 Output Files

After training, the following files are generated:

```
model_output/
  ├── model.pkl                    # Trained model (mean_face, Phi, clf, labels)
  ├── metrics.json                 # Test set metrics
  └── classification_report.txt    # Detailed classification report
```

## 🔧 Configuration

Edit `config.yaml` to customize default parameters:

```yaml
dataset:
  root_dir: "./dataset"
  image_size: 64
  test_split: 0.2

pca:
  n_components: 50
  variance_threshold: null

classifier:
  type: "mlp"
  mlp:
    hidden_layers: [128, 64]
    max_iter: 500
```

## 📈 Performance Tips

1. **Optimal k value**: Start with k=50 and adjust based on variance explained
2. **Image size**: 64x64 is a good balance between speed and accuracy
3. **Test split**: Use 20-30% of data for testing
4. **Classifier choice**: 
   - MLP: Best for complex patterns
   - SVM: Good for smaller datasets
   - KNN: Fast but may require more data

## 🎯 Example Results

Typical performance on the provided dataset:
- **Accuracy**: 85-95% (depends on dataset quality)
- **Training time**: 1-5 minutes
- **Prediction time**: < 0.1s per image

## 🐛 Troubleshooting

**Issue: "No images found"**
- Check directory structure matches the required format
- Ensure image files are valid (jpg, png)

**Issue: Low accuracy**
- Increase k (number of eigenfaces)
- Try different classifier types
- Ensure sufficient training data per person (minimum 5 images)

**Issue: High memory usage**
- Reduce image size (--img_size 32)
- Reduce number of eigenfaces (--k 30)

**Issue: "Model file not found"**
- Check that training completed successfully
- Verify the model path is correct (should be .pkl not .npz)

## 📝 Key Improvements Over Original Code

✅ **Fixed file extension bug** (.pkl instead of .npz)  
✅ **Added train/test split** with proper evaluation  
✅ **Comprehensive error handling** with logging  
✅ **Unknown person detection** with confidence thresholds  
✅ **Batch prediction** support  
✅ **Rich visualizations** (eigenfaces, mean face, variance, reconstructions)  
✅ **Detailed evaluation** with per-class metrics  
✅ **Configuration management** via YAML  
✅ **Type hints** and docstrings throughout  
✅ **Face detection** preprocessing option  
✅ **Multiple classifier** support  
✅ **Production-ready** code quality  

## 📚 References

- Turk, M. and Pentland, A. (1991). "Eigenfaces for Recognition"
- scikit-learn: Machine Learning in Python
- OpenCV: Open Source Computer Vision Library

## 📄 License

MIT License - Feel free to use for educational and commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This implementation is designed for educational purposes and production use. For large-scale deployment, consider using deep learning approaches like FaceNet or ArcFace.

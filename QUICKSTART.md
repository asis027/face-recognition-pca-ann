# Quick Start Guide

This guide will help you get started with the face recognition system in under 5 minutes!

## 📋 Prerequisites

- Python 3.7 or higher
- 2GB RAM minimum
- Dataset of face images

## 🚀 Installation (2 minutes)

### Option 1: Automatic Setup (Linux/Mac)

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup (All platforms)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📁 Prepare Your Dataset (1 minute)

Organize images like this:

```
dataset/
  ├── alice/
  │   ├── photo1.jpg
  │   ├── photo2.jpg
  │   └── photo3.jpg
  ├── bob/
  │   ├── photo1.jpg
  │   └── photo2.jpg
  └── charlie/
      └── photo1.jpg
```

**Tips:**
- Each person needs their own folder
- Use at least 5 images per person for best results
- Images should be reasonably clear and contain faces
- Any common image format works (jpg, png, etc.)

## 🎓 Train Your Model (1 minute)

```bash
python train.py --data_dir ./dataset
```

This will:
- Split data into training (80%) and testing (20%)
- Train the model
- Show test accuracy
- Save model to `model_output/model.pkl`

**Expected output:**
```
Loading images from ./dataset
Loaded 15 images for alice
Loaded 12 images for bob
...
Test Accuracy: 92.50%
Model saved to model_output/model.pkl
```

## 🔮 Make Predictions (30 seconds)

### Single Image

```bash
python predict.py \
  --model model_output/model.pkl \
  --image path/to/test_photo.jpg
```

**Output:**
```
==================================================
Predicted Person: alice
Confidence: 94.23%
Status: Known Person
==================================================
```

### Multiple Images

```bash
# Create list of images
echo "photo1.jpg" > images.txt
echo "photo2.jpg" >> images.txt

# Predict all at once
python predict.py \
  --model model_output/model.pkl \
  --batch images.txt
```

## 📊 Common Use Cases

### Case 1: High Accuracy Training

```bash
python train.py \
  --data_dir ./dataset \
  --k 100 \
  --classifier mlp \
  --hidden_layers 256 128 64 \
  --img_size 128
```

### Case 2: Fast Training (for testing)

```bash
python train.py \
  --data_dir ./dataset \
  --k 20 \
  --classifier knn \
  --img_size 32
```

### Case 3: Security System (strict matching)

```bash
python predict.py \
  --model model_output/model.pkl \
  --image door_camera.jpg \
  --confidence_threshold 0.8
```

### Case 4: Photo Organization (lenient matching)

```bash
python predict.py \
  --model model_output/model.pkl \
  --batch family_photos.txt \
  --confidence_threshold 0.3
```

## 🎨 Visualize Results

```bash
# See what the AI "sees"
python visualize.py \
  --model model_output/model.pkl \
  --eigenfaces \
  --mean_face \
  --save
```

This creates:
- `visualizations/eigenfaces.png` - The "ghost faces" used for recognition
- `visualizations/mean_face.png` - Average of all faces

## 📈 Evaluate Performance

```bash
python evaluate.py \
  --model model_output/model.pkl \
  --data_dir ./dataset \
  --plots
```

Generates detailed reports showing:
- Overall accuracy
- Per-person accuracy
- Common mistakes
- Confidence analysis

## ⚡ Performance Tips

| Goal | Setting |
|------|---------|
| Best accuracy | `--k 100 --img_size 128 --classifier mlp` |
| Fastest training | `--k 20 --img_size 32 --classifier knn` |
| Balanced | `--k 50 --img_size 64 --classifier mlp` (default) |
| Small dataset | `--k 10 --classifier svm` |
| Large dataset | `--k 100 --classifier mlp` |

## 🐛 Troubleshooting

### "No images found"
- Check folder structure: `dataset/person_name/image.jpg`
- Ensure images are valid (not corrupted)
- Check file permissions

### Low accuracy (<70%)
- Add more images per person (aim for 10+)
- Increase k: `--k 100`
- Increase image size: `--img_size 128`
- Check image quality

### "Out of memory"
- Reduce image size: `--img_size 32`
- Reduce k: `--k 20`
- Use smaller batches

### Prediction says "Unknown" for known person
- Lower threshold: `--confidence_threshold 0.3`
- Check if person was in training data
- Add more training images for that person

## 📚 Next Steps

1. **Read the full documentation**: `README.md`
2. **Customize settings**: Edit `config.yaml`
3. **Run tests**: `python -m pytest test_pca_ann.py -v`
4. **Explore advanced features**: Face detection, batch processing, etc.

## 💡 Pro Tips

1. **More training images = better accuracy**
   - Aim for 10-20 images per person
   - Include variety: different angles, lighting, expressions

2. **Image quality matters**
   - Use clear, well-lit photos
   - Face should be clearly visible
   - Avoid extreme angles or occlusions

3. **Tune the confidence threshold**
   - Higher (0.7-0.9): Strict, fewer false positives
   - Lower (0.3-0.5): Lenient, fewer false negatives
   - Default (0.5): Balanced

4. **Regular retraining**
   - Add new photos over time
   - Retrain when adding new people
   - Keep test accuracy above 85%

## 🆘 Need Help?

- Check README.md for detailed documentation
- Review error messages carefully
- Test with a smaller dataset first
- Make sure all dependencies are installed

---

**You're all set! Start recognizing faces! 🎉**

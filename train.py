"""
Prediction script for face recognition system.
Supports single image prediction with confidence scores and unknown face detection.
"""

import argparse
import joblib
import numpy as np
import cv2
import os
import logging
from typing import Dict, Tuple

from pca_ann import project_faces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Dict:
    """
    Load the trained model and components.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model components
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    data = joblib.load(model_path)
    
    required_keys = ['mean_face', 'Phi', 'labels', 'clf', 'image_size']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Model file is missing required key: {key}")
    
    logger.info("Model loaded successfully!")
    return data


def preprocess_image(
    image_path: str,
    image_size: int,
    detect_face: bool = False
) -> np.ndarray:
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        image_size: Target size for the image
        detect_face: Whether to detect face in the image
        
    Returns:
        Preprocessed image as flattened array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Optional: Face detection
    if detect_face:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            logger.warning("No face detected in image. Processing entire image.")
        else:
            # Use the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            img = img[y:y+h, x:x+w]
            logger.info(f"Face detected at ({x}, {y}, {w}, {h})")
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    img = cv2.resize(img, (image_size, image_size))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img.flatten()


def predict_with_confidence(
    clf,
    X_proj: np.ndarray,
    labels: Dict[int, str],
    threshold: float = 0.5
) -> Tuple[str, float, bool]:
    """
    Make prediction with confidence score and unknown detection.
    
    Args:
        clf: Trained classifier
        X_proj: Projected face features
        labels: Label mapping dictionary
        threshold: Confidence threshold for unknown detection
        
    Returns:
        Tuple of (predicted_label, confidence, is_known)
    """
    pred = clf.predict(X_proj)[0]
    predicted_label = labels[pred]
    
    # Get confidence if available
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X_proj)[0]
        confidence = np.max(proba)
        
        # Check if confidence is below threshold (unknown person)
        if confidence < threshold:
            return "Unknown", confidence, False
    else:
        confidence = 1.0  # For classifiers without probability estimates
    
    return predicted_label, confidence, True


def predict_batch(
    model_data: Dict,
    image_paths: list,
    confidence_threshold: float = 0.5,
    detect_face: bool = False
):
    """
    Predict multiple images at once.
    
    Args:
        model_data: Loaded model dictionary
        image_paths: List of image paths
        confidence_threshold: Threshold for unknown detection
        detect_face: Whether to detect faces
    """
    mean_face = model_data['mean_face']
    Phi = model_data['Phi']
    labels = model_data['labels']
    clf = model_data['clf']
    image_size = model_data['image_size']
    
    results = []
    
    for img_path in image_paths:
        try:
            img = preprocess_image(img_path, image_size, detect_face)
            X_proj = project_faces(Phi, mean_face, img.reshape(1, -1))
            pred_label, confidence, is_known = predict_with_confidence(
                clf, X_proj, labels, confidence_threshold
            )
            
            results.append({
                'image': img_path,
                'prediction': pred_label,
                'confidence': confidence,
                'is_known': is_known
            })
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            results.append({
                'image': img_path,
                'prediction': 'Error',
                'confidence': 0.0,
                'is_known': False,
                'error': str(e)
            })
    
    return results


def main(args):
    """Main prediction function."""
    
    # Load model
    model_data = load_model(args.model)
    
    mean_face = model_data['mean_face']
    Phi = model_data['Phi']
    labels = model_data['labels']
    clf = model_data['clf']
    image_size = model_data['image_size']
    
    logger.info(f"Model info: {len(labels)} classes, {Phi.shape[1]} eigenfaces")
    logger.info(f"Classes: {', '.join(labels.values())}")
    
    # Process single image or batch
    if args.batch:
        if not os.path.exists(args.batch):
            logger.error(f"Batch file not found: {args.batch}")
            return
        
        with open(args.batch, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(image_paths)} images...")
        results = predict_batch(
            model_data,
            image_paths,
            args.confidence_threshold,
            args.detect_face
        )
        
        # Print results
        print("\n" + "="*70)
        print(f"{'Image':<30} {'Prediction':<20} {'Confidence':<10} {'Status'}")
        print("="*70)
        for r in results:
            status = "Known" if r['is_known'] else "Unknown"
            if 'error' in r:
                status = "Error"
            print(f"{os.path.basename(r['image']):<30} {r['prediction']:<20} "
                  f"{r['confidence']*100:>6.2f}%    {status}")
        print("="*70)
        
    else:
        # Single image prediction
        try:
            img = preprocess_image(args.image, image_size, args.detect_face)
            X_proj = project_faces(Phi, mean_face, img.reshape(1, -1))
            
            pred_label, confidence, is_known = predict_with_confidence(
                clf, X_proj, labels, args.confidence_threshold
            )
            
            print("\n" + "="*50)
            print(f"Predicted Person: {pred_label}")
            print(f"Confidence: {confidence*100:.2f}%")
            print(f"Status: {'Known Person' if is_known else 'Unknown Person'}")
            print("="*50 + "\n")
            
            if not is_known:
                logger.warning(
                    f"Low confidence ({confidence*100:.2f}%). "
                    f"This person may not be in the training set."
                )
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict face identity using trained model'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model file (model.pkl)'
    )
    parser.add_argument(
        '--image',
        help='Path to test image (for single prediction)'
    )
    parser.add_argument(
        '--batch',
        help='Path to text file with image paths (one per line)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for unknown detection (default: 0.5)'
    )
    parser.add_argument(
        '--detect_face',
        action='store_true',
        help='Enable face detection preprocessing'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be provided")
    
    main(args)

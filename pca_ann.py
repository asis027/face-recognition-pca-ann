"""
PCA and Eigenface Implementation for Face Recognition
Provides utilities for PCA computation, eigenface generation, and face projection.
"""

import numpy as np
import cv2
import os
from typing import Tuple, Dict, Optional
from sklearn.utils import shuffle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_images_from_folder(
    root_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    gray: bool = True,
    test_split: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load images from folder structure and split into train/test sets.
    
    Args:
        root_dir: Root directory containing person folders
        image_size: Target size for resizing images
        gray: Whether to convert to grayscale
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, labels dictionary
    """
    X, y, labels = [], [], {}
    label_idx = 0
    total_images = 0
    failed_images = 0
    
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")
    
    logger.info(f"Loading images from {root_dir}")
    
    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
            
        labels[label_idx] = person
        person_images = 0
        
        for fname in sorted(os.listdir(person_path)):
            fpath = os.path.join(person_path, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    logger.warning(f"Could not read image: {fpath}")
                    failed_images += 1
                    continue
                    
                if gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                img = cv2.resize(img, image_size)
                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0
                X.append(img.flatten())
                y.append(label_idx)
                person_images += 1
                total_images += 1
                
            except Exception as e:
                logger.warning(f"Error processing {fpath}: {e}")
                failed_images += 1
                continue
        
        logger.info(f"Loaded {person_images} images for {person}")
        label_idx += 1
    
    if len(X) == 0:
        raise ValueError("No images loaded. Check directory structure.")
    
    logger.info(f"Total images loaded: {total_images}")
    logger.info(f"Failed images: {failed_images}")
    logger.info(f"Number of classes: {len(labels)}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle and split
    X, y = shuffle(X, y, random_state=random_state)
    
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, labels


def compute_mean_face(X: np.ndarray) -> np.ndarray:
    """
    Compute the mean face from a set of face images.
    
    Args:
        X: Array of shape (n_samples, n_features)
        
    Returns:
        Mean face vector of shape (n_features,)
    """
    return np.mean(X, axis=0)


def mean_normalize(X: np.ndarray, mean_face: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean face.
    
    Args:
        X: Array of shape (n_samples, n_features)
        mean_face: Mean face vector of shape (n_features,)
        
    Returns:
        Centered data of shape (n_samples, n_features)
    """
    return X - mean_face


def compute_eigenfaces(
    Delta: np.ndarray,
    k: int,
    variance_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenfaces using the surrogate covariance method.
    
    Args:
        Delta: Mean-centered face matrix of shape (n_samples, n_features)
        k: Number of eigenfaces to compute
        variance_threshold: Optional threshold for cumulative variance explained
        
    Returns:
        Phi: Eigenfaces matrix of shape (n_features, k)
        eigvals: Top k eigenvalues
        explained_variance_ratio: Variance explained by each component
    """
    n_samples = Delta.shape[0]
    
    if k > n_samples:
        logger.warning(f"k={k} > n_samples={n_samples}. Setting k={n_samples}")
        k = n_samples
    
    # Compute surrogate covariance matrix (n_samples x n_samples)
    C = (1.0 / n_samples) * np.dot(Delta, Delta.T)
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(C)
    
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigvals)
    explained_variance_ratio = eigvals / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Adjust k based on variance threshold if provided
    if variance_threshold is not None:
        k_var = np.argmax(cumulative_variance >= variance_threshold) + 1
        logger.info(f"Components needed for {variance_threshold*100}% variance: {k_var}")
        k = min(k, k_var)
    
    # Take top k eigenvectors
    eigvecs_k = eigvecs[:, :k]
    
    # Compute eigenfaces: Delta^T * eigvecs_k
    Phi = np.dot(Delta.T, eigvecs_k)
    
    # Normalize eigenfaces
    for i in range(Phi.shape[1]):
        norm = np.linalg.norm(Phi[:, i])
        if norm > 1e-8:
            Phi[:, i] /= norm
    
    logger.info(f"Using {k} eigenfaces")
    logger.info(f"Cumulative variance explained: {cumulative_variance[k-1]*100:.2f}%")
    
    return Phi, eigvals[:k], explained_variance_ratio[:k]


def project_faces(
    Phi: np.ndarray,
    mean_face: np.ndarray,
    X: np.ndarray
) -> np.ndarray:
    """
    Project faces onto eigenface space.
    
    Args:
        Phi: Eigenfaces matrix of shape (n_features, k)
        mean_face: Mean face vector of shape (n_features,)
        X: Face images of shape (n_samples, n_features)
        
    Returns:
        Projected faces of shape (n_samples, k)
    """
    Xc = X - mean_face
    return np.dot(Xc, Phi)


def reconstruct_face(
    weights: np.ndarray,
    Phi: np.ndarray,
    mean_face: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a face from its eigenface weights.
    
    Args:
        weights: Eigenface weights of shape (k,)
        Phi: Eigenfaces matrix of shape (n_features, k)
        mean_face: Mean face vector of shape (n_features,)
        
    Returns:
        Reconstructed face of shape (n_features,)
    """
    return mean_face + np.dot(Phi, weights)

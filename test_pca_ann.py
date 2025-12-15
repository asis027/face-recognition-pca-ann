"""
Unit tests for PCA and face recognition components.
Run with: python -m pytest test_pca_ann.py -v
"""

import numpy as np
import pytest
import tempfile
import os
import cv2
from pca_ann import (
    compute_mean_face,
    mean_normalize,
    compute_eigenfaces,
    project_faces,
    reconstruct_face
)


@pytest.fixture
def sample_faces():
    """Generate sample face data for testing."""
    np.random.seed(42)
    n_samples = 20
    n_features = 64 * 64
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 5, n_samples)
    return X, y


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with sample images."""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample face images
    for person_id in range(3):
        person_dir = os.path.join(temp_dir, f"person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)
        
        for img_id in range(5):
            # Create random grayscale image
            img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            img_path = os.path.join(person_dir, f"img_{img_id}.jpg")
            cv2.imwrite(img_path, img)
    
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestPCAComponents:
    """Test PCA computation components."""
    
    def test_compute_mean_face(self, sample_faces):
        """Test mean face computation."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        
        assert mean_face.shape == (X.shape[1],)
        assert np.allclose(mean_face, np.mean(X, axis=0))
    
    def test_mean_normalize(self, sample_faces):
        """Test mean normalization."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        X_normalized = mean_normalize(X, mean_face)
        
        assert X_normalized.shape == X.shape
        assert np.allclose(np.mean(X_normalized, axis=0), 0, atol=1e-6)
    
    def test_compute_eigenfaces(self, sample_faces):
        """Test eigenface computation."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        k = 10
        Phi, eigvals, variance_ratio = compute_eigenfaces(Delta, k)
        
        # Check shapes
        assert Phi.shape == (X.shape[1], k)
        assert len(eigvals) == k
        assert len(variance_ratio) == k
        
        # Check eigenfaces are normalized
        for i in range(k):
            norm = np.linalg.norm(Phi[:, i])
            assert np.isclose(norm, 1.0, atol=1e-6)
        
        # Check variance ratios sum to <= 1
        assert np.sum(variance_ratio) <= 1.0
    
    def test_eigenfaces_k_too_large(self, sample_faces):
        """Test eigenface computation when k > n_samples."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        k = 100  # Larger than n_samples
        Phi, eigvals, variance_ratio = compute_eigenfaces(Delta, k)
        
        # Should automatically adjust k to n_samples
        assert Phi.shape[1] <= X.shape[0]
    
    def test_project_faces(self, sample_faces):
        """Test face projection to eigenspace."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        k = 10
        Phi, _, _ = compute_eigenfaces(Delta, k)
        
        X_proj = project_faces(Phi, mean_face, X)
        
        assert X_proj.shape == (X.shape[0], k)
    
    def test_reconstruct_face(self, sample_faces):
        """Test face reconstruction from eigenspace."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        k = 10
        Phi, _, _ = compute_eigenfaces(Delta, k)
        
        # Project and reconstruct first face
        X_proj = project_faces(Phi, mean_face, X[0:1])
        reconstructed = reconstruct_face(X_proj[0], Phi, mean_face)
        
        assert reconstructed.shape == X[0].shape
        
        # Reconstruction should be similar to original
        # (not exact due to dimensionality reduction)
        mse = np.mean((X[0] - reconstructed) ** 2)
        assert mse < 1.0  # Reasonable reconstruction error


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self):
        """Test with single sample (edge case)."""
        X = np.random.rand(1, 100).astype(np.float32)
        mean_face = compute_mean_face(X)
        
        assert mean_face.shape == (100,)
        assert np.allclose(mean_face, X[0])
    
    def test_zero_variance_feature(self):
        """Test with zero variance feature."""
        X = np.random.rand(10, 100).astype(np.float32)
        X[:, 0] = 1.0  # Zero variance in first feature
        
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        # Should still work without errors
        Phi, _, _ = compute_eigenfaces(Delta, 5)
        assert Phi.shape == (100, 5)
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        X = np.random.rand(10, 10000).astype(np.float32)
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        # Should handle large feature space
        Phi, _, _ = compute_eigenfaces(Delta, 5)
        assert Phi.shape == (10000, 5)


class TestNumericalStability:
    """Test numerical stability of computations."""
    
    def test_orthogonality_of_eigenfaces(self, sample_faces):
        """Test that eigenfaces are orthogonal."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        k = 5
        Phi, _, _ = compute_eigenfaces(Delta, k)
        
        # Check orthogonality: Phi^T * Phi should be close to identity
        orthogonality = np.dot(Phi.T, Phi)
        identity = np.eye(k)
        
        assert np.allclose(orthogonality, identity, atol=1e-5)
    
    def test_variance_ordering(self, sample_faces):
        """Test that eigenvalues are in descending order."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        Phi, eigvals, _ = compute_eigenfaces(Delta, 10)
        
        # Eigenvalues should be in descending order
        assert np.all(eigvals[:-1] >= eigvals[1:])
    
    def test_projection_invertibility(self, sample_faces):
        """Test that projection + reconstruction approximately recovers original."""
        X, _ = sample_faces
        mean_face = compute_mean_face(X)
        Delta = mean_normalize(X, mean_face)
        
        # Use all components for near-perfect reconstruction
        k = min(X.shape[0], X.shape[1])
        Phi, _, _ = compute_eigenfaces(Delta, k)
        
        X_proj = project_faces(Phi, mean_face, X[0:1])
        reconstructed = reconstruct_face(X_proj[0], Phi, mean_face)
        
        # Should be very close to original
        mse = np.mean((X[0] - reconstructed) ** 2)
        assert mse < 0.01


class TestDataTypes:
    """Test handling of different data types."""
    
    def test_float32_input(self):
        """Test with float32 input."""
        X = np.random.rand(10, 100).astype(np.float32)
        mean_face = compute_mean_face(X)
        assert mean_face.dtype == np.float32
    
    def test_float64_input(self):
        """Test with float64 input."""
        X = np.random.rand(10, 100).astype(np.float64)
        mean_face = compute_mean_face(X)
        # Should work with float64
        assert mean_face.shape == (100,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

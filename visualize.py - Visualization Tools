"""
Visualization tools for eigenfaces, mean face, and reconstructions.
"""

import argparse
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from pca_ann import reconstruct_face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_eigenfaces(Phi, image_size, n_components=16, save_path=None):
    """
    Visualize the top eigenfaces.
    
    Args:
        Phi: Eigenfaces matrix of shape (n_features, k)
        image_size: Size of each face image
        n_components: Number of eigenfaces to display
        save_path: Optional path to save the figure
    """
    n_components = min(n_components, Phi.shape[1])
    n_rows = int(np.ceil(np.sqrt(n_components)))
    n_cols = int(np.ceil(n_components / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle('Top Eigenfaces', fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_components > 1 else [axes]
    
    for i in range(n_components):
        eigenface = Phi[:, i].reshape(image_size, image_size)
        
        # Normalize for visualization
        eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Eigenfaces saved to {save_path}")
    
    plt.show()


def visualize_mean_face(mean_face, image_size, save_path=None):
    """
    Visualize the mean face.
    
    Args:
        mean_face: Mean face vector
        image_size: Size of the face image
        save_path: Optional path to save the figure
    """
    mean_img = mean_face.reshape(image_size, image_size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_img, cmap='gray')
    plt.title('Mean Face', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Mean face saved to {save_path}")
    
    plt.show()


def visualize_variance_explained(variance_ratio, save_path=None):
    """
    Plot the variance explained by each principal component.
    
    Args:
        variance_ratio: Array of variance ratios
        save_path: Optional path to save the figure
    """
    cumulative_variance = np.cumsum(variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linewidth=2, markersize=4)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Variance plot saved to {save_path}")
    
    plt.show()


def visualize_reconstruction(
    original_face,
    Phi,
    mean_face,
    image_size,
    n_components_list=[5, 10, 20, 50],
    save_path=None
):
    """
    Visualize face reconstruction with different numbers of components.
    
    Args:
        original_face: Original face vector
        Phi: Eigenfaces matrix
        mean_face: Mean face vector
        image_size: Size of the face image
        n_components_list: List of component counts to try
        save_path: Optional path to save the figure
    """
    n_plots = len(n_components_list) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Original face
    original_img = original_face.reshape(image_size, image_size)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructions with different k
    for idx, k in enumerate(n_components_list, 1):
        if k > Phi.shape[1]:
            k = Phi.shape[1]
        
        # Project to k-dimensional space
        Phi_k = Phi[:, :k]
        weights = np.dot(original_face - mean_face, Phi_k)
        
        # Reconstruct
        reconstructed = reconstruct_face(weights, Phi_k, mean_face)
        reconstructed_img = reconstructed.reshape(image_size, image_size)
        
        # Calculate reconstruction error
        mse = np.mean((original_face - reconstructed) ** 2)
        
        axes[idx].imshow(reconstructed_img, cmap='gray')
        axes[idx].set_title(f'k={k}\nMSE={mse:.4f}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reconstruction comparison saved to {save_path}")
    
    plt.show()


def visualize_confusion_matrix(confusion_matrix, labels, save_path=None):
    """
    Visualize confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        labels: Label mapping dictionary
        save_path: Optional path to save the figure
    """
    label_names = [labels[i] for i in sorted(labels.keys())]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def main(args):
    """Main visualization function."""
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    model_data = joblib.load(args.model)
    
    mean_face = model_data['mean_face']
    Phi = model_data['Phi']
    labels = model_data['labels']
    image_size = model_data['image_size']
    
    # Create output directory
    output_dir = args.output or 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize based on user selection
    if args.all or args.eigenfaces:
        logger.info("Visualizing eigenfaces...")
        save_path = os.path.join(output_dir, 'eigenfaces.png') if args.save else None
        visualize_eigenfaces(Phi, image_size, args.n_eigenfaces, save_path)
    
    if args.all or args.mean_face:
        logger.info("Visualizing mean face...")
        save_path = os.path.join(output_dir, 'mean_face.png') if args.save else None
        visualize_mean_face(mean_face, image_size, save_path)
    
    if args.all or args.variance:
        if 'variance_ratio' in model_data:
            logger.info("Visualizing variance explained...")
            save_path = os.path.join(output_dir, 'variance_explained.png') if args.save else None
            visualize_variance_explained(model_data['variance_ratio'], save_path)
        else:
            logger.warning("Variance ratio not found in model. Skipping variance plot.")
    
    if args.reconstruction:
        if not os.path.exists(args.reconstruction):
            logger.error(f"Reconstruction image not found: {args.reconstruction}")
            return
        
        logger.info("Visualizing reconstruction...")
        img = cv2.imread(args.reconstruction, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32) / 255.0
        
        save_path = os.path.join(output_dir, 'reconstruction.png') if args.save else None
        visualize_reconstruction(
            img.flatten(),
            Phi,
            mean_face,
            image_size,
            args.k_values,
            save_path
        )
    
    if args.confusion_matrix:
        # Load confusion matrix from metrics
        metrics_path = os.path.join(os.path.dirname(args.model), 'metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            logger.info("Visualizing confusion matrix...")
            save_path = os.path.join(output_dir, 'confusion_matrix.png') if args.save else None
            visualize_confusion_matrix(
                np.array(metrics['confusion_matrix']),
                labels,
                save_path
            )
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")
    
    logger.info(f"Visualizations complete! Output saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize PCA components and model performance'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model file (model.pkl)'
    )
    parser.add_argument(
        '--output',
        help='Output directory for saving visualizations'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all visualizations'
    )
    parser.add_argument(
        '--eigenfaces',
        action='store_true',
        help='Visualize eigenfaces'
    )
    parser.add_argument(
        '--n_eigenfaces',
        type=int,
        default=16,
        help='Number of eigenfaces to display (default: 16)'
    )
    parser.add_argument(
        '--mean_face',
        action='store_true',
        help='Visualize mean face'
    )
    parser.add_argument(
        '--variance',
        action='store_true',
        help='Plot variance explained'
    )
    parser.add_argument(
        '--reconstruction',
        help='Path to image for reconstruction visualization'
    )
    parser.add_argument(
        '--k_values',
        nargs='+',
        type=int,
        default=[5, 10, 20, 50],
        help='k values for reconstruction (default: 5 10 20 50)'
    )
    parser.add_argument(
        '--confusion_matrix',
        action='store_true',
        help='Visualize confusion matrix'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualizations to files'
    )
    
    args = parser.parse_args()
    
    if not any([args.all, args.eigenfaces, args.mean_face, args.variance, 
                args.reconstruction, args.confusion_matrix]):
        parser.error("At least one visualization option must be specified")
    
    main(args)

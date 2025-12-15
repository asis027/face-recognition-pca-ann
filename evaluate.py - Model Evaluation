"""
Comprehensive model evaluation script.
Provides detailed metrics, per-class analysis, and error analysis.
"""

import argparse
import joblib
import numpy as np
import cv2
import os
import json
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from pca_ann import load_images_from_folder, project_faces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_on_dataset(model_data, X_test, y_test):
    """
    Evaluate model on test dataset.
    
    Args:
        model_data: Loaded model dictionary
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with comprehensive metrics
    """
    mean_face = model_data['mean_face']
    Phi = model_data['Phi']
    clf = model_data['clf']
    labels = model_data['labels']
    
    # Project test faces
    X_test_proj = project_faces(Phi, mean_face, X_test)
    
    # Predictions
    y_pred = clf.predict(X_test_proj)
    
    # Get probabilities if available
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_proj)
        confidences = np.max(y_proba, axis=1)
    else:
        confidences = np.ones(len(y_pred))
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_test,
        'confidences': confidences,
        'labels': labels
    }


def analyze_errors(results):
    """
    Analyze prediction errors in detail.
    
    Args:
        results: Dictionary from evaluate_on_dataset
        
    Returns:
        Dictionary with error analysis
    """
    y_true = results['true_labels']
    y_pred = results['predictions']
    confidences = results['confidences']
    labels = results['labels']
    
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    error_analysis = {
        'total_errors': len(misclassified_idx),
        'error_rate': len(misclassified_idx) / len(y_true),
        'errors': []
    }
    
    for idx in misclassified_idx:
        error_analysis['errors'].append({
            'true_label': labels[y_true[idx]],
            'predicted_label': labels[y_pred[idx]],
            'confidence': float(confidences[idx])
        })
    
    # Most common confusion pairs
    confusion_pairs = {}
    for idx in misclassified_idx:
        pair = (labels[y_true[idx]], labels[y_pred[idx]])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    most_confused = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    error_analysis['most_confused_pairs'] = [
        {'true': pair[0], 'predicted': pair[1], 'count': count}
        for (pair, count) in most_confused[:10]
    ]
    
    # Low confidence predictions
    low_conf_threshold = 0.6
    low_conf_idx = np.where(confidences < low_conf_threshold)[0]
    error_analysis['low_confidence_predictions'] = len(low_conf_idx)
    error_analysis['low_confidence_rate'] = len(low_conf_idx) / len(confidences)
    
    return error_analysis


def plot_per_class_metrics(results, save_path=None):
    """
    Plot per-class performance metrics.
    
    Args:
        results: Dictionary from evaluate_on_dataset
        save_path: Optional path to save the figure
    """
    labels = results['labels']
    label_names = [labels[i] for i in sorted(labels.keys())]
    
    precision = results['precision_per_class']
    recall = results['recall_per_class']
    f1 = results['f1_per_class']
    accuracy = results['per_class_accuracy']
    
    x = np.arange(len(label_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, f1, width, label='F1-Score', alpha=0.8)
    ax.bar(x + 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    plt.show()


def plot_confidence_distribution(results, save_path=None):
    """
    Plot distribution of prediction confidences.
    
    Args:
        results: Dictionary from evaluate_on_dataset
        save_path: Optional path to save the figure
    """
    confidences = results['confidences']
    y_true = results['true_labels']
    y_pred = results['predictions']
    
    correct_mask = y_true == y_pred
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(correct_conf, bins=30, alpha=0.6, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=30, alpha=0.6, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(
        [correct_conf, incorrect_conf],
        labels=['Correct', 'Incorrect'],
        showmeans=True
    )
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Confidence by Prediction Correctness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence distribution plot saved to {save_path}")
    
    plt.show()


def generate_report(results, error_analysis, output_dir):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Dictionary from evaluate_on_dataset
        error_analysis: Dictionary from analyze_errors
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FACE RECOGNITION MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:  {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {results['precision']*100:.2f}%\n")
        f.write(f"Recall:    {results['recall']*100:.2f}%\n")
        f.write(f"F1-Score:  {results['f1_score']*100:.2f}%\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*70 + "\n")
        labels = results['labels']
        for i in sorted(labels.keys()):
            f.write(f"\n{labels[i]}:\n")
            f.write(f"  Accuracy:  {results['per_class_accuracy'][i]*100:.2f}%\n")
            f.write(f"  Precision: {results['precision_per_class'][i]*100:.2f}%\n")
            f.write(f"  Recall:    {results['recall_per_class'][i]*100:.2f}%\n")
            f.write(f"  F1-Score:  {results['f1_per_class'][i]*100:.2f}%\n")
            f.write(f"  Support:   {results['support_per_class'][i]}\n")
        
        # Error analysis
        f.write("\n" + "="*70 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Errors: {error_analysis['total_errors']}\n")
        f.write(f"Error Rate:   {error_analysis['error_rate']*100:.2f}%\n\n")
        
        f.write("Most Confused Pairs:\n")
        for pair in error_analysis['most_confused_pairs'][:5]:
            f.write(f"  {pair['true']} → {pair['predicted']}: {pair['count']} times\n")
        
        f.write(f"\nLow Confidence Predictions: {error_analysis['low_confidence_predictions']}\n")
        f.write(f"Low Confidence Rate: {error_analysis['low_confidence_rate']*100:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    logger.info(f"Evaluation report saved to {report_path}")


def main(args):
    """Main evaluation function."""
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    model_data = joblib.load(args.model)
    
    # Load test dataset
    if args.test_dir:
        logger.info(f"Loading test dataset from {args.test_dir}...")
        X_test, _, y_test, _, labels = load_images_from_folder(
            args.test_dir,
            image_size=(model_data['image_size'], model_data['image_size']),
            gray=True,
            test_split=0.0,
            random_state=42
        )
        model_data['labels'] = labels
    else:
        # Use the test split from training
        logger.info("Loading original dataset for evaluation...")
        X_train, X_test, y_train, y_test, labels = load_images_from_folder(
            args.data_dir,
            image_size=(model_data['image_size'], model_data['image_size']),
            gray=True,
            test_split=0.2,
            random_state=42
        )
        model_data['labels'] = labels
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_on_dataset(model_data, X_test, y_test)
    
    # Error analysis
    logger.info("Analyzing errors...")
    error_analysis = analyze_errors(results)
    
    # Create output directory
    output_dir = args.output or 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    generate_report(results, error_analysis, output_dir)
    
    # Save detailed metrics as JSON
    metrics_path = os.path.join(output_dir, 'detailed_metrics.json')
    with open(metrics_path, 'w') as f:
        json_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'error_analysis': error_analysis
        }
        json.dump(json_results, f, indent=2)
    
    # Generate plots
    if args.plots:
        logger.info("Generating plots...")
        plot_per_class_metrics(
            results,
            os.path.join(output_dir, 'per_class_metrics.png')
        )
        
        if hasattr(model_data['clf'], 'predict_proba'):
            plot_confidence_distribution(
                results,
                os.path.join(output_dir, 'confidence_distribution.png')
            )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Low Confidence Predictions: {error_analysis['low_confidence_predictions']}")
    print("\nTop Confused Pairs:")
    for pair in error_analysis['most_confused_pairs'][:3]:
        print(f"  {pair['true']} → {pair['predicted']}: {pair['count']} times")
    print("="*70)
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive model evaluation'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model file (model.pkl)'
    )
    parser.add_argument(
        '--data_dir',
        help='Original dataset directory (for test split)'
    )
    parser.add_argument(
        '--test_dir',
        help='Separate test dataset directory'
    )
    parser.add_argument(
        '--output',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.test_dir:
        parser.error("Either --data_dir or --test_dir must be provided")
    
    main(args)

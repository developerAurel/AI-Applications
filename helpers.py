"""
helpers.py — Shared utility functions for AiAp Miniproject 1
Authors: Aurel Köppel, Yves Fricker

This module provides reusable functions for:
- Loading and preprocessing the Animals-10 dataset
- Splitting data into train/val/test sets
- Plotting learning curves and confusion matrices
- Computing classification metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# Mapping: Italian folder names → English class names
CLASS_MAPPING = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}


def load_animals10(
    data_dir,
    img_size=64,
    samples_per_class=1000,
    reduced_class="elefante",
    reduced_count=150,
    seed=42,
):
    """
    Load the Animals-10 dataset from a local directory.

    The dataset is expected to have subdirectories named in Italian
    (e.g., 'cane', 'gatto', ...), each containing image files.

    Parameters
    ----------
    data_dir : str
        Path to the root directory of the Animals-10 dataset.
    img_size : int
        Target size for resizing images (img_size x img_size).
    samples_per_class : int
        Maximum number of samples to keep per class.
    reduced_class : str
        Italian name of the class to reduce (for imbalanced data task).
    reduced_count : int
        Number of samples to keep for the reduced class.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Image data with shape (n_samples, img_size, img_size, 3), values in [0, 1].
    y : np.ndarray
        Integer labels with shape (n_samples,).
    class_names : list of str
        English class names, ordered by label index.
    """
    rng = np.random.default_rng(seed)

    # Sort folder names for consistent label ordering
    folder_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    images = []
    labels = []
    class_names = []
    label_idx = 0

    for folder in folder_names:
        if folder not in CLASS_MAPPING:
            continue

        folder_path = os.path.join(data_dir, folder)
        english_name = CLASS_MAPPING[folder]
        class_names.append(english_name)

        # Collect all valid image file paths
        file_list = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        rng.shuffle(file_list)

        # Determine how many samples to keep
        if folder == reduced_class:
            max_samples = reduced_count
        else:
            max_samples = samples_per_class

        count = 0
        for fname in file_list:
            if count >= max_samples:
                break
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((img_size, img_size))
                images.append(np.array(img))
                labels.append(label_idx)
                count += 1
            except Exception:
                # Skip corrupted images
                continue

        print(f"  Loaded {count} images for class '{english_name}' (folder: {folder})")
        label_idx += 1

    X = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y = np.array(labels, dtype=np.int32)

    print(f"\nTotal: {len(X)} images, {len(class_names)} classes")
    return X, y, class_names


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split data into train+validation and test sets using stratified sampling.

    Parameters
    ----------
    X : np.ndarray
        Image data.
    y : np.ndarray
        Labels.
    test_size : float
        Fraction of data to reserve for testing.
    random_state : int
        Random seed.

    Returns
    -------
    X_trainval, X_test, y_trainval, y_test : np.ndarray
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train+Val: {len(X_trainval)} samples")
    print(f"Test:      {len(X_test)} samples (locked away)")
    return X_trainval, X_test, y_trainval, y_test


def plot_learning_curves(history, title="Learning Curves"):
    """
    Plot training and validation loss/accuracy from a Keras history object.

    Shows 4 curves:
    - Training Loss
    - Validation Loss
    - Training Accuracy
    - Validation Accuracy

    Parameters
    ----------
    history : keras.callbacks.History
        The history object returned by model.fit().
    title : str
        Title for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss
    ax1.plot(epochs, history.history["loss"], "b-", label="Training Loss")
    ax1.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"], "b-", label="Training Accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(y_true, y_pred, class_names, title_prefix=""):
    """
    Plot 4 confusion matrices with different normalization options.

    The 4 variants are:
    - normalize=None:   absolute counts
    - normalize='true': normalized by true labels (rows sum to 1)
    - normalize='pred': normalized by predicted labels (columns sum to 1)
    - normalize='all':  normalized by total number of samples

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Class names for axis labels.
    title_prefix : str
        Prefix for plot titles.
    """
    normalize_options = [None, "true", "pred", "all"]
    titles = [
        "Absolute Counts",
        "Normalized by True Label (Recall)",
        "Normalized by Predicted Label (Precision)",
        "Normalized by Total Samples",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, norm, subtitle in zip(axes.flat, normalize_options, titles):
        cm = confusion_matrix(y_true, y_pred, normalize=norm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format=".0f" if norm is None else ".2f")
        ax.set_title(f"{title_prefix}{subtitle}")
        # Rotate x-axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def print_classification_metrics(y_true, y_pred, class_names):
    """
    Print Precision, Recall, and F1-score for each class.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str
        Class names.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("Classification Report:")
    print("=" * 60)
    print(report)

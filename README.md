# Skin Cancer Detection

## Project Objective

Skin cancer is one of the most prevalent and potentially deadly forms of cancer. Traditional diagnostic methods can be subjective, resource-intensive, and time-consuming. This project aims to develop an Machine Learning solution to accurately classify skin lesions as **Malignant** (cancerous) or **Benign** (non-cancerous), improving detection accuracy and accessibility.

## Dataset: HAM10000

The project utilizes the **HAM10000** dataset, consisting of dermoscopic images of skin lesions.

*   **Original Size:** 10,015 images
*   **Classes:**
    *   **Benign:** Nevus, Dermatofibroma, Benign Keratosis-like Lesion
    *   **Malignant:** Basal Cell Carcinoma, Actinic Keratosis, Melanoma
*   **Preprocessing & Augmentation:**
    *   Images resized to **224 x 224**.
    *   Data split: **Train (70%)**, **Validation (15%)**, **Test (15%)**.
    *   **Augmentation:** Applied to address class imbalance (2x for benign, 5x for cancerous). Techniques include horizontal flip, rotation, contrast adjustment, zoom, and translation.
    *   **Final Distribution:**
        *   Training: 18,124 images
        *   Validation: 1,502 images
        *   Test: 1,503 images

## Code Structure

The final model implementations are:

*   **`CNN.ipynb`**: **Convolutional Neural Network (CNN)**
    *   Simple architecture with 2 hidden layers.
    *   Uses ReduceLROnPlateau and Early Stopping.
*   **`ResNet50.ipynb`**: **ResNet-50**
    *   Deep residual network pretrained on ImageNet.
    *   Uses class weights to handle imbalance.
*   **`swin.ipynb`**: **Swin Transformer**
    *   Hierarchical Transformer using shifted windows.
*   **`vit.ipynb`**: **Vision Transformer (ViT)**
    *   Transformer-based architecture using global self-attention.

## Key Results

The following table summarizes the performance of the evaluated models (based on project report):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | 61% | 69% | 48% | 53% | 50% |
| **ResNet-50** | 85% | 84% | 85% | 85% | 89% |
| **ViT** | 83% | 86% | 83% | 84% | 91% |
| **Swin** | 84% | 68% | 67% | 68% | 88% |

**Key Findings:**
*   **Transformers (ViT & Swin)** showed excellent ROC-AUC scores but are computationally intensive.
*   **ResNet-50** provided strong overall performance with high accuracy and recall.

## Requirements

*   **Python 3.x**
*   **TensorFlow / Keras** (for CNN, ResNet50)
*   **PyTorch, torchvision, timm** (for Swin, ViT)
*   `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

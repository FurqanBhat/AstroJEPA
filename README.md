# 🌌 AstroJEPA: Cosmic Self-Supervised Learning with LeJEPA

> **🎓 Capstone Project Status:** Active & Ongoing (2025)

**AstroJEPA** is a research initiative applying advanced Self-Supervised Learning (SSL) to astronomical imaging. The goal is to learn robust semantic representations of galaxy morphologies from unlabeled data, enabling downstream tasks like anomaly detection, redshift estimation, and galaxy classification without heavy reliance on human annotation.

## 🔭 Project Overview

Unlike autoencoders (which reconstruct pixels) or contrastive methods (which often require heavy data augmentation inappropriate for scientific data), **AstroJEPA** attempts to learn a high-level world model of cosmic structures by predicting the **latent representations** of masked image regions.

This project explores the transition from standard Joint-Embedding architectures to the mathematically grounded **LeJEPA** framework to solve stability issues common in low-data regimes.

## 📓 Notebooks & Workflow

This repository is structured around two core Jupyter Notebooks that handle the lifecycle of the model:

### 1. `training_astrojepa_300_epochs.ipynb`
This notebook handles the heavy lifting of the self-supervised learning process.
*   **Data Pipeline:** Implements the custom `Galaxy10FullDataset` and the brightness-biased masking collator.
*   **Model Definition:** Defines the Context Encoder, Target Encoder, and Predictor (ViT architecture).
*   **Training Loop:** Implements the optimization logic.
*   **Output:** Saves model checkpoints to disk.

### 2. `inference_astro_jepa_300_epochs.ipynb`
This notebook is used for evaluating the quality of the learned representations. It includes a comprehensive suite of analysis tools:
*   **Latent Visualization:** Inspects how the model "sees" galaxies by comparing predicted vs. actual latent patches using cosine similarity and MSE heatmaps.
*   **Anomaly Detection:** Computes anomaly scores based on the model's prediction error, identifying galaxies that the model finds "surprising" (potentially rare morphological features).
*   **Dimensionality Reduction:** Uses t-SNE to project the high-dimensional latent space into 2D, colored by galaxy class or anomaly score.
*   **Unsupervised Clustering:** Applies K-Means clustering on the learned embeddings to discover inherent groupings within the galaxy data and analyzes cluster purity against ground-truth labels.

## 🧠 Methodology & Architecture

### 1. The Backbone (Mini-JEPA)
The core architecture processes 224x224 galaxy images using a Vision Transformer (ViT) backbone:
*   **Context Encoder:** Processes the visible parts of the galaxy image.
*   **Target Encoder:** Processes the masked block to create a "ground truth" latent vector.
*   **Predictor:** A lightweight Transformer that takes the Context output and a positional mask token to predict the output of the Target Encoder.

### 2. Galaxy-Aware Masking
Astronomical images are sparse (mostly background noise). Standard random masking fails to capture semantic meaning. We implemented a custom **Brightness-Biased Semantic Masking** strategy:
*   The collator computes patch brightness to identify high-signal regions (the galaxy).
*   **Context blocks** are sampled from these high-signal regions to ensure the model actually sees the galaxy.
*   **Target blocks** are masked regions of the galaxy that the model is forced to predict based on the context.

## 🚀 The Challenge & Solution

In the early stages, we implemented the standard I-JEPA loss and observed **Representation Collapse**—where the model outputs constant vectors to minimize loss without learning features.

To solve this, we pivoted to **LeJEPA (Latent-Euclidean JEPA)** based on Balestriero & LeCun (2025). We implemented the **SIGReg (Sketched Isotropic Gaussian Regularization)** loss, which mathematically ensures the embeddings utilize the full latent space, minimizing downstream prediction risk and preventing collapse.

## 📦 Model Zoo

This repository contains three distinct model checkpoints representing the evolution of our research:

| Model Filename | Architecture | Training Duration | Methodology | Status |
| :--- | :--- | :--- | :--- | :--- |
| `100jepa.pth` | Mini-JEPA | 100 Epochs | Standard I-JEPA | **Baseline.** Shows early signs of feature collapse. |
| `300cosmic_jepa_latest.pth` | Mini-JEPA | 300 Epochs | Standard I-JEPA | **Poor Training.** Low loss, but suffers from severe collapse. |
| `100lejepa.pth` | **LeJEPA** | 100 Epochs | **SIGReg + Prediction** | **Current Best.** Uses LeJEPA regularization to maintain feature diversity. |

## 📂 Dataset

We utilize the **Galaxy10 DECals** dataset for training and evaluation:
*   **Source:** DESI Legacy Imaging Surveys.
*   **Size:** 17,736 images.
*   **Format:** 256x256 pixel images (Center-cropped to 224x224).

## 🛠️ Usage & Analysis

### Inference Tools
The inference notebook provides several powerful functions for analyzing model performance:

*   `visualize_full_latent_map(model, dataset, idx)`: Generates side-by-side plots of the original galaxy, a heatmap of cosine similarity between predicted and target patches, and a heatmap of MSE.
*   `compute_jepa_anomaly_scores(model, dataset)`: iterates through the dataset and returns a Pandas DataFrame containing anomaly scores, allowing for easy sorting and filtering of rare galaxies.
*   `extract_latents(model, dataset)`: Efficiently processes the entire dataset to return a numpy array of latent vectors for downstream tasks.
*   `kmeans_torch(X, num_clusters)`: A GPU-accelerated implementation of K-Means to cluster the extracted latent vectors.

### Visualization
*   **t-SNE:** We use t-SNE to visualize the structure of the learned manifold.
*   **Cluster Analysis:** Heatmaps showing the relationship between unsupervised clusters and ground-truth galaxy classes.

## 📚 References

*   **LeJEPA Paper:** Balestriero, R., & LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. arXiv:2511.08544.
*   **I-JEPA:** Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*.

## 🤝 Contributing

This is an academic capstone project. Contributions regarding the stability of SSL in low-data regimes (like astronomy) are welcome.

**License:** MIT License.

---
*Created by [Furqan Bhat] - Capstone Project 2025*
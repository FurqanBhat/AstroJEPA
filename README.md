# 🌌 AstroJEPA: Latent-Euclidean Self-Supervised Learning for Galaxy Morphology Analysis

> **🎓 Capstone Project Status:** Active & Ongoing (2025)  
> **Framework:** Joint-Embedding Predictive Architectures (JEPA) with Sketched Isotropic Gaussian Regularization (SIGReg)  
> **Author:** Furqan Bhat | Computer Engineering Department

---

## Executive Summary

**AstroJEPA** is a rigorous investigation into **self-supervised representation learning** for astronomical imaging, specifically focusing on the stability and semantic quality of **Joint-Embedding Predictive Architectures (JEPAs)** in low-data astronomical regimes. This project explores the theoretical and empirical transition from standard **Image JEPA (I-JEPA)** to the mathematically grounded **Latent-Euclidean JEPA (LeJEPA)** framework, implementing **Sketched Isotropic Gaussian Regularization (SIGReg)** to solve representation collapse—a persistent challenge in SSL for small scientific datasets.

We have trained **5 distinct JEPA models** across parameter scales (5.5M and 25.3M parameters) and regularization configurations, achieving competitive downstream performance on **Galaxy10 morphology classification**, **unsupervised anomaly detection**, and **latent space visualization tasks**.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Model Zoo & Training Configurations](#model-zoo--training-configurations)
4. [Dataset](#dataset)
5. [Methodology & Architecture](#methodology--architecture)
6. [Experimental Results](#experimental-results)
7. [Usage & Inference](#usage--inference)
8. [Key Findings & Discussion](#key-findings--discussion)
9. [References](#references)

---

## 🔭 Project Overview

### Motivation

Unlike natural image datasets where hand-crafted augmentations (cropping, color jittering) preserve semantic information, **astronomical images present a unique challenge**: the galaxy signal is sparse, spatially heterogeneous, and scientifically valuable *only in its original form*. Traditional SSL methods (contrastive approaches, generative methods) either:

- **Fail on small datasets**: Require millions of samples for stable training
- **Introduce harmful biases**: Hand-crafted augmentations obscure subtle morphological features
- **Suffer from collapse**: Joint-embedding methods degenerate to constant representations without heuristic safeguards

**AstroJEPA** addresses these challenges by:

1. **Adopting JEPA philosophy**: Predicting *representation-space outputs* rather than pixels, eliminating need for augmentations
2. **Implementing SIGReg regularization**: Enforcing isotropic Gaussian embeddings; the theoretical optimal distribution that minimizes downstream task risk
3. **Replacing heuristics with theory**: Removing teacher-student networks, stop-gradients, and asymmetric architectures in favor of mathematically principled design


### Core Contribution

This project instantiates **LeJEPA** (Balestriero & LeCun, 2025) for astronomical data and provides:

- **Five trained JEPA models** with systematic ablations (I-JEPA, I-JEPA with SIGReg intensity variations, Le-JEPA model is still being worked upon)
- **Comprehensive evaluation suite**: Loss curves, anomaly detection, linear probes, latent space visualizations
- **Evidence that principled SSL scales** to edge-compute astronomical pipelines

---

## 🧬 Theoretical Foundations

### Joint-Embedding Predictive Architectures (JEPA)

A **JEPA** trains an encoder \(\text{Enc}\) to make embeddings predictable under semantic relationships, formalized as:

\[
\mathcal{L}_{\text{pred}} = \mathbb{E}_{(v_1, v_2) \in \mathcal{V}} \left[ \left\| \text{Pred}\left(\text{Enc}(v_1), m_2\right) - \text{Enc}(v_2) \right\|_2^2 \right]
\]

where \(v_1, v_2\) are semantic views (e.g., context and target image blocks), \(m_2\) is a positional mask token, and **Pred** is a lightweight transformer predictor.

### The Representation Collapse Problem

Early JEPA implementations exhibit **representation collapse**: encoders output near-constant embeddings—because the prediction loss alone admits trivial solutions. Standard fixes (stop-gradient, EMA teacher networks, asymmetric architectures) are heuristic and unstable.

### LeJEPA: Principled Regularization via SIGReg

**LeJEPA** solves collapse by enforcing that embeddings follow an **isotropic Gaussian distribution**, which provably minimizes worst-case downstream task risk:

**Theorem (Balestriero & LeCun, 2025):** Among all distributions with fixed variance, the isotropic Gaussian \(\mathcal{N}(0, I)\) uniquely minimizes integrated square bias for both linear and non-linear probes.

**Sketched Isotropic Gaussian Regularization (SIGReg)** operationalizes this via univariate statistical tests projected along random directions:

\[
\mathcal{L}_{\text{SIGReg}} = \frac{1}{|A|} \sum_{a \in A} \text{EppsPulley}\left( z \cdot a, \mathcal{N}(0, 1) \right)
\]

where \(\text{EppsPulley}\) compares empirical characteristic functions and \(A\) is a set of random projection directions.

**Key benefits:**
- Linear time/memory complexity in dimension and batch size
- Theoretically bounded gradients (Theorem 4)
- Overcomes curse of dimensionality via Sobolev smoothness (Theorem 5)
- Single hyperparameter \(\lambda\) controlling prediction-vs.-regularization trade-off

---

## 📦 Model Zoo & Training Configurations

### Overview Table

| Model ID | Architecture | Parameters | Training Epochs | Loss Configuration | Lambda (λ) SIGReg | Linear Probe Accuacy | Status |
|:---|:---|:---:|:---:|:---|:---:|:---|:---|
| **Model 1** | ViT-Base (5.5M) | 5.5M | 100 | Pure I-JEPA | — | 30% | ✓ Trained |
| **Model 2** | ViT-Base (5.5M) | 5.5M | 300 | Pure I-JEPA | — | 40% | ✓ Trained |
| **Model 3** | ViT-Base (5.5M) | 5.5M | 300 | I-JEPA + SIGReg | 0.6 (strong) | 50% | ✓ Trained |
| **Model 4** | ViT-Small (25.3M) | 25.3M | 100 | I-JEPA + SIGReg | 0.6 (strong) | 60% | ✓ Trained |
| **Model 5** | ViT-Small (25.3M) | 25.3M | 100 | I-JEPA + Slight SIGReg | 0.01 (faint) | 55% | ✓ Trained |

```

### Prediction Loss Details

All models use the standard JEPA L2 prediction loss:

\[
\mathcal{L}_{\text{pred}} = \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i} \left\| \hat{s}_{y_j} - s_{y_j} \right\|_2^2
\]

where \(M=4\) target blocks per image, \(\hat{s}_{y_j}\) is predicted latent and \(s_{y_j}\) is target-encoder output.

### SIGReg Configuration (Models 3–5)

| Hyperparameter | Value | Justification |
|:---|:---:|:---|
| Statistical Test | Epps-Pulley (characteristic function) | Stable gradients, no sorting overhead |
| Random Projections per step | 256 | Empirically sufficient for ~2K-dim embeddings |
| Integration domain | [-5, 5] | Characteristic function evaluation range |
| Quadrature points | 17 (trapezoidal rule) | Balances accuracy and efficiency |
| Resampling | Per minibatch | Cumulative coverage effect beats fixed directions |

---

## 📊 Dataset

### Galaxy10 DECals

- **Source:** DESI Legacy Imaging Surveys (DECals)
- **Total Images:** 17,736 (training split)
- **Classes:** 10 morphological categories (elliptical, spiral, irregular, etc.)
- **Image Format:** 256×256 RGB, center-cropped to 224×224 during training
- **Preprocessing:**
  - Standardization: per-image z-score normalization (zero-mean, unit variance)
  - No aggressive augmentation (preserves scientific signal)
  - Brightness-biased mask generation: patches computed via mean pixel intensity

### Brightness-Biased Masking (Galaxy-Specific)

Standard random masking fails for sparse astronomical images. Our custom collator:

1. **Computes patch brightness** \(b_k\) for each 16×16 patch
2. **Identifies "signal patches"**: high-brightness regions indicating galaxy presence
3. **Context sampling**: draws from signal patches to ensure model sees galaxy
4. **Target sampling**: independently samples 4 blocks from full image
5. **Overlap removal**: strips overlapping regions from context to maintain non-trivial prediction

This ensures the model learns galaxy-specific features rather than background noise.

---

## 🏗️ Methodology & Architecture

### Network Components

```
┌─────────────────────────────────────────────────────────────────┐
│ Input Image (224×224) - Galaxy from Galaxy10 Dataset             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼ (Context Block)                    ▼ (Target Block)
   ┌─────────────┐                      ┌─────────────┐
   │  Context    │                      │   Target    │
   │  Encoder    │                      │   Encoder   │
   │   (ViT)     │                      │    (ViT)    │
   │ Frozen EMA  │                      │  Frozen EMA │
   └──────┬──────┘                      └──────┬──────┘
          │ s_x (context embedding)            │ s_y (target embedding)
          ▼                                    │
   ┌─────────────────────┐                    │
   │   Predictor (ViT)   │◄─── Positional    │
   │   (4 target blocks) │     Mask Tokens   │
   └──────┬──────────────┘                    │
          │ ŝ_y (predicted target)            │
          │                                   │
          └───────────┬──────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
    ┌─────────────┐         ┌──────────────┐
    │ L2 Distance │         │  SIGReg Loss │
    │  (Pred vs   │         │  (Gaussian   │
    │   Target)   │         │   Match)     │
    └──────┬──────┘         └───────┬──────┘
           │                        │
           └────────────┬───────────┘
                        ▼
          LeJEPA Loss = (1-λ) L_pred + λ L_SIGReg
                        (λ ∈ {0, 0.01, 0.6})
```

### Architecture Specifications

| Component | Configuration | Details |
|:---|:---|:---|
| **Context/Target Encoder** | Vision Transformer (ViT) | 12 layers, 768 hidden dims (ViT-S), 16×16 patch embedding, depth=12 |
| **Predictor** | Lightweight ViT | 6 layers (half encoder depth), conditioned on positional mask tokens |
| **Embedding Dimension** | 768 (ViT) or higher | For 25.3M param model: 1024-dim embeddings |
| **Latent Space** | L2-normalized after SIGReg | Optional normalization for downstream tasks |
| **Target Encoder Update** | EMA (τ=0.999) | \(\theta_{\text{target}} \leftarrow 0.999 \cdot \theta_{\text{target}} + 0.001 \cdot \theta_{\text{context}}\) |

### Loss Functions in Detail

#### Prediction Loss
\[
\mathcal{L}_{\text{pred}} = \mathbb{E}_{b=1}^{B} \frac{1}{4} \sum_{i=1}^{4} \left\| \text{Pred}(s_x, m_i) - s_{y_i} \right\|_2^2
\]

#### SIGReg Loss (Models 3–5)
\[
\mathcal{L}_{\text{SIGReg}} = \frac{1}{256} \sum_{a \in A} \left| \int_{-5}^{5} \left( \phi_{\text{emp}}(t) - \phi_{\mathcal{N}(0,1)}(t) \right)^2 w(t) dt \right|
\]

where \(\phi_{\text{emp}}(t) = \frac{1}{N} \sum_{i=1}^{N} e^{i t (z_i \cdot a)}\) is the empirical characteristic function and \(w(t) = e^{-t^2/2}\) is Gaussian weighting.

#### Total LeJEPA Loss
\[
\mathcal{L}_{\text{LeJEPA}} = (1 - \lambda) \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{SIGReg}}
\]

---

## 📈 Experimental Results


### Model 2: I-JEPA Extended (5.5M params, 300 epochs)
**File:** `300cosmic_jepa_latest.pth`

**Training Dynamics:**
- Loss decreases drastically, then increases and then finally decreases again but plateaus after epoch 150
- Severe collapse confirmed by constant near-zero cosine similarities in latent space
- Final training loss: [0.0705]
- Final validation loss: [0.0334]

**Loss Curve:**
![Model2 Loss](outputs/model2/model2_loss.png)

**Anomaly Detection (Top 25):**
![](outputs/model2/model2_top25anomalies.png)

**Linear Probe Score:** [40] % (better than Model 1)

**Latent Space Visualization (t-SNE):**
![](outputs/model3/model3_latentspace.png)

**Cosine Similarity Heatmaps (Sample Galaxy):**
![Predicted vs. Target Similarity Heatmap | MSE Heatmap](outputs/model2/model2_patch_mse.png)

**Key Observation:** Extended training without regularization improves probe accuracy but it's still low.

---

### Model 3: I-JEPA + SIGReg Hybrid (5.5M params, 300 epochs, λ=0.6)
**File:** `100ijepa_segreg.pth`

**Training Dynamics:**
- Final training loss: [0.1279]
- Final Validation loss: [0.0594]

**Loss Curve (Prediction vs. SIGReg):**
![](outputs/model3/model3_loss.png)

**Anomaly Detection (Top 25):**
![](outputs/model3/model3_top25anomalies.png)

**Linear Probe Score:** [52] % (Significant improvement over Models 1 & 2)

**Latent Space Visualization (t-SNE):**
![](outputs/model3/model3_latentspace.png)

**Cosine Similarity Heatmaps (Sample Galaxy):**
![](outputs/model3/model3_patch_mse.png)

**Key Observation:** Hybrid approach stabilizes training and preserves semantic information. SIGReg contribution clear.

---

### Model 4: LeJEPA Full (25.3M params, 100 epochs, λ=0.6)
**File:** `25.3M_100EP_ViT_strong_SIGReg.pth`

**Training Dynamics:**
- Training loss decreases drastically while validation loss increases for first 20 epochs implying overfitting.
- After epoch 20, both training and validation loss star decreasing and the loss plateaus around epoch 65
- Loss exhibits strong correlation with downstream performance.
- Larger model benefits from SIGReg; Better Accuracy detected.
- Final training loss: [0.1252]
- Final validation loss: [0.0337]
- Final SIGReg loss: [0.1112]

**Loss Curve:**
![](outputs/model4/model4_loss.png)

**Anomaly Detection (Top 25):**
![](outputs/model4/model4_top25anomalies.png)

**Linear Probe Score (Frozen Backbone):** [59.13] % Top-1


**Latent Space Visualization (PCA):**
![](outputs/model4/model4_latentspace.png)

**Cosine Similarity Heatmaps (Sample Galaxy):**
![](outputs/model4/model4_patch_mse.png)

**Key Observations:**
- Highest linear probe accuracy among all models
- Rich latent structure visible in PCA
- Anomaly scores identify rare morphologies (mergers, distorted galaxies)

---

### Model 5: LeJEPA Minimal SIGReg (25.3M params, 100 epochs, λ=0.01)
**File:** `25.3M_100EP_ViT_slight_SIGReg.pth`

**Training Dynamics:**
- Low SIGReg weight (λ=0.01) reduces regularization strength
- Prediction loss dominates (≈99% of total loss)
- Final training loss: [0.0523]
- Final validation loss: [0.1112]
- Final SIGReg loss: [0.1536]

**Loss Curve:**
![](outputs/model5/model5_loss.png)

**Anomaly Detection (Top 25):**
![](outputs/model5/model5_top25anomalies.png)

**Linear Probe Score (Frozen Backbone):** [57.1] %

**Latent Space Visualization (PCA):**
![](outputs/model5/model5_latentspace.png)

**Cosine Similarity Heatmaps (Sample Galaxy):**
![](outputs/model5/model5_patch_mse.png)

**Key Observation:** λ=0.01 works alright but slightly highier values of lambda could perform better. Demonstrates necessity of stronger regularization.

---




## 🔬 Key Findings & Discussion

### 1. SIGReg Stabilizes Training & Prevents Collapse

**Observation:** Models 1–2 (pure I-JEPA) exhibit training loss convergence masking representation collapse, evidenced by:
- Constant near-zero cosine similarities between predicted and target blocks
- Poor linear probe performance despite low training loss
- Uniform latent space lacking class structure

**Models 3–5** with SIGReg show:
- Smooth, stable training curves
- Training loss strongly correlates with downstream accuracy
- Rich, well-separated latent space (visible in t-SNE/PCA)

**Conclusion:** SIGReg mathematically enforces embedding isotropy, eliminating collapse by design rather than heuristic safeguards.

---

### 2. λ-Sensitivity: Stronger Regularization ≠ Always Better, But Minimum Threshold Exists

| Model | λ | Downstream Accuracy | Conclusion |
|:---|:---:|:---:|:---|
| 1 | — | [Val] % | Pure prediction, collapse |
| 2 | — | [Val] % | Extended training, worse |
| 3 | 0.6 | [Val] % | Balanced, good performance |
| 4 | 0.6 | [Val] % | **Best overall** |
| 5 | 0.01 | [Val] % | Too weak; subtle collapse |

**Insight:** λ=0.6 provides strong regularization while maintaining prediction fidelity. λ=0.01 insufficient. Sweet spot appears \(\lambda \in [0.3, 0.8]\).

---

### 3. Parameter Scaling: LeJEPA Scales Gracefully to 25.3M

- **5.5M models (Models 1–3)**: Collapse risk high without regularization
- **25.3M models (Models 4–5)**: Larger capacity stabilizes training, but SIGReg remains essential
- **Recommendation:** For astronomical SSL, 5.5M ViT-S is sufficient; 25.3M offers marginal gains

---


### 5. Anomaly Detection Reveals Morphological Diversity

Top 10 anomalous galaxies consistently include:
- **Galaxy mergers** (high interaction energy)
- **Ultra-luminous infrared galaxies** (rare in Galaxy10)
- **Distorted/tidal tails** (asymmetric morphologies)
- **Ultra-compact dwarfs** (extreme morphology)

**Utility:** Anomaly scores from [Model 4] effectively identify astrophysically interesting outliers without supervised labels.



---

## 📚 References

### Primary Papers

1. **Balestriero, R., & LeCun, Y. (2025).** LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv:2511.08544*.

2. **Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023).** Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

### Related Work

3. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *CVPR*.

4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *ICCV*.

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.

### Astronomical Datasets

6. Zhou, R., Knobati, C., Geach, J. E., & Gall, C. (2021). Galaxy10 DECals: A large sample galaxy morphology dataset. *The Astrophysical Journal Supplement Series, 255*(1), 1–18.

---


## 📄 License

This project is released under the **MIT License**. See `LICENSE` file for details.

---

## 👤 Author

**Furqan Bhat**  
Computer Science Department | Capstone Project (2025)  
[Abdullah Gul University]

For questions or collaboration: [furqan.eec@gmail.com]

---


**Last Updated:** December 2025  
**Status:** ✅ Ongoing (5 models trained and evaluated)

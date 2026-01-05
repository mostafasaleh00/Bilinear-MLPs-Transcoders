# Neural Network Interpretability Project

This project explores interpretability techniques for MLPs trained on MNIST, comparing two approaches: **Bilinear MLPs** and **Transcoders**.

## Notebooks

### 1. `bilinear_mlp_mnist.ipynb`

Implements the paper "Bilinear MLPs Enable Weight-Based Mechanistic Interpretability".

**Key Concept:** A bilinear layer computes `g(x) = (Wx) * (Vx)` (element-wise product, no nonlinearity), which can be rewritten as a quadratic form `g(x) = x^T B x`. This allows eigendecomposition to reveal interpretable features.

**Contents:**
- Bilinear layer implementation
- Interaction matrices and bilinear tensor computation
- Eigendecomposition analysis for each digit class
- Visualization of eigenvectors as digit-specific feature detectors
- Low-rank approximation (10-20 eigenvectors capture most accuracy)
- Adversarial mask construction from weights alone
- Comparison of models with/without noise regularization

**Results:**
- Test accuracy: ~98%
- 10 eigenvectors/class: ~97% accuracy
- 20 eigenvectors/class: ~98% accuracy (near full performance)

### 2. `transcoder_interpretability.ipynb`

Uses sparse autoencoders (transcoders) to interpret a standard ReLU MLP.

**Key Concept:** A transcoder learns to predict MLP layer outputs from inputs using a sparse, overcomplete representation. The sparse features can be backprojected to pixel space for visualization.

**Contents:**
- Standard MLP training (784 -> 512 -> 10)
- Transcoder training (784 -> 2048 sparse -> 512)
- Feature visualization via backprojection
- Class-specific feature analysis
- Feature importance spectra (analogous to eigenvalues)
- Misclassification analysis using feature activations
- t-SNE visualization of sparse feature space
- Low-rank structure analysis

**Results:**
- MLP test accuracy: ~98%
- Transcoder exhibits low-rank structure similar to bilinear MLPs
- 32-64 active features capture most model behavior

## Comparison

| Aspect | Bilinear MLP | Standard MLP + Transcoder |
|--------|--------------|---------------------------|
| Interpretability Method | Eigendecomposition | Sparse Autoencoder |
| Features | Eigenvectors | Encoder weights |
| Importance | Eigenvalues | Decoder weights |
| Low-rank structure | Yes | Yes |
| Requires extra training | No | Yes (transcoder) |
| Works with standard activations | No | Yes |

## Requirements

- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn (for t-SNE)
- tqdm

## Usage

Run either notebook in Jupyter or JupyterLab. Both will download MNIST automatically on first run.

```bash
jupyter notebook bilinear_mlp_mnist.ipynb
# or
jupyter notebook transcoder_interpretability.ipynb
```

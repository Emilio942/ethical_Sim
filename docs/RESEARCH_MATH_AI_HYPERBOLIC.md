# Research Note: Hyperbolic Manifolds for Ethical Latent Spaces
**Source:** Mathematik-KI
**Date:** April 13, 2026

## Core Proposal
Transition the latent space $\mathcal{M}$ from Euclidean $\mathbb{R}^n$ to a **Hyperbolic Manifold** $\mathbb{H}^n$ using the **Poincaré Ball model**.

## Mathematical Foundation
- **Metric Tensor:** $g_{ij}(\mathbf{x}) = \frac{4}{(1-\|\mathbf{x}\|^2)^2}\delta_{ij}$
- **Geodesic Distance:** $d_{\mathbb{H}}(\mathbf{u},\mathbf{v}) = \operatorname{arcosh}\left(1+ \frac{2\|\mathbf{u}-\mathbf{v}\|^2}{(1-\|\mathbf{u}\|^2)(1-\|\mathbf{v}\|^2)}\right)$
- **Attention weights:** $\alpha_{i}(\mathbf{z}) = \frac{\exp(-\beta d_{\mathbb{H}}(\mathbf{z},\mathbf{p}_i))}{\sum_{j} \exp(-\beta d_{\mathbb{H}}(\mathbf{z},\mathbf{p}_j))}$

## Key Benefits
1. **Hierarchical Capacity:** Exponential growth of volume with radius allows for natural embedding of ethical taxonomies.
2. **Sharper Gating:** Hyperbolic distances grow sub-linearly with Euclidean separation, leading to more discrete "selection" of ethical prototypes.
3. **Riemannian Gradient Descent (R-GD):** Accelerates updates for points deep in the hierarchy (near the boundary) while maintaining stability.

## Implementation Strategy
- Use **Exp-map** at the origin for initial projection.
- Implement **Retractions** to keep points within the unit ball.
- Apply **Metric-aware learning rates**.

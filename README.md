# Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting

> *"I am seeking exaggeration in the essential."* — Vincent van Gogh, 1888

This repository contains the official implementation of **"Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting"**, a flow-guided geometric advection framework that computationally realizes Post-Impressionist principles by prioritizing directional syntax over texture projection.

## Abstract

In 1888, Vincent van Gogh articulated: *"I am seeking exaggeration in the essential."* This principle—magnifying structural form while eliminating photographic detail—defined Post-Impressionist art. However, current 3D style transfer methods invert this philosophy, treating geometry as a rigid canvas for texture projection. To authentically replicate Post-Impressionist stylization, we must embrace geometric abstraction as the mechanism of expression.

We present a flow-guided geometric advection framework for 3D Gaussian Splatting (3DGS) that extracts 2D directional flow from paintings and back-propagates these patterns to rectify 3D Gaussians, producing flow-aligned brushstrokes that wrap around scene topology without mesh priors.

## Key Features

- 🎨 **Flow-Guided Geometric Advection**: Mesh-free flow guidance via projection analysis that "combs" 3DGS primitives into directional brushstrokes
- 🌊 **Luminance-Structure Decoupling**: Separates geometric deformation from color optimization, preventing artifacts during structural abstraction
- 🖼️ **VLM-as-a-Judge Framework**: Assesses artistic authenticity through aesthetic judgment rather than pixel metrics
- 🎭 **Post-Impressionist Stylization**: Achieves structural abstraction characteristic of Post-Impressionism by sacrificing photographic fidelity for expressive coherence

## Method Overview

Our framework reconstructs the volumetric brushstrokes of master artists (e.g., Van Gogh) through a **Geometry-First, Color-Second** strategy:

1. **Flow-Aware Primitive Rectification**: Extracts dominant local stroke orientation from style reference using structure tensor analysis
2. **Gradient-Driven Advection Optimization**: Back-propagates 2D flow gradients to rectify 3D Gaussian positions and rotations
3. **Luminance-Structure Decoupling**: Optimizes geometric flow in luminance space while maintaining chromatic consistency

The optimization jointly minimizes three types of energies:
- **Flow-alignment energy**: Enforces coherent, anisotropic brushstroke geometry
- **Geometric regularization energy**: Preserves global 3D structure while allowing controlled deformation
- **Appearance decoupling energy**: Separates structural stylization from chromatic consistency

## Installation

### Requirements
- NVIDIA GPU with CUDA 11.8+
- Python 3.10
- PyTorch 2.3.0

### Setup

```bash
# Clone repository with submodules
git clone https://github.com/zhendong-zdw/TLVG-GS --recursive
cd TLVG-GS

# Install Python dependencies
conda env create -f environment.yaml
conda activate TLVG-GS

# Install submodules
pip install tlvg/submodules/lang-segment-anything
pip install gs/submodules/diff-gaussian-rasterization
pip install gs/submodules/simple-knn
```

## Quick Start

### Dataset Preparation
- **Scene datasets**: LLFF dataset from [NeRF](https://github.com/bmild/nerf) and T&T dataset from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- **Style images**: Post-Impressionist masterpieces (e.g., Van Gogh's *The Starry Night*, Munch's *The Scream*)

**Important**: For optimal stylization results, ensure that the original scene is trained using 0th-order spherical harmonics (SH) coefficients. Higher-order SH coefficients may introduce artifacts during the style transfer process.

### Single Style Transfer

```bash
python style_transfer.py --config configs/llff_single.yaml
python style_transfer.py --config configs/tnt_single.yaml
```

### Semantic-aware Style Transfer

```bash
python style_transfer.py --config configs/llff_semantic.yaml
```

### Compositional Style Transfer

```bash
python style_transfer.py --config configs/llff_compositional.yaml
```

## Results

Our method produces constitutive brushstrokes that physically rotate and align with the scene's curvature, creating coherent painterly flow that mimics the artist's hand. This results in a strong sense of relief and directional energy absent in prior works.

![Qualitative Results](./tlvg/assets/fig6.png)

### Key Advantages

- **Geometric Flow**: Brushstrokes wrap around scene topology, avoiding "flat sticker" artifacts
- **Volumetric Relief**: Creates impasto effects through geometric deformation
- **Artistic Authenticity**: Prioritizes aesthetic energy over physical accuracy

## Evaluation

We evaluate our method using:

1. **VLM-as-a-Judge**: A panel of AI critics (GPT-5.1, GPT-4o, Claude 4.5, Claude 3.5, Gork, Qwen 3) performing randomized pairwise comparisons
2. **User Studies**: 30 participants (18 art experts, 12 laypeople) evaluating Flow Alignment, Painterly Materiality, and Aesthetic Preference

Our method achieves:
- **Win Rate**: >85% across geometric metrics
- **Average Authenticity Score**: 8.36/10 (vs. 7.19 for baseline)

## Acknowledgements

This project builds upon the following works:

- **3D Gaussian Splatting**: The core 3D Gaussian rendering and optimization framework is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Bernhard Kerbl et al.
- **ABC-GS**: Our implementation builds upon [ABC-GS](https://github.com/vpx-ecnu/ABC-GS) for the base framework
- **Lang-Segment-Anything**: Semantic segmentation functionality powered by [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)

## License

See [LICENSE.md](LICENSE.md) for details.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

*"What critics derided as 'crude' proved revolutionary—inspiring Post-Impressionism and Expressionism to embrace structural syntax over literal representation."*

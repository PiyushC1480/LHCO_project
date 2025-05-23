# Neural Networks for LHC Dataset

## Overview

This project focuses on distinguishing hadronic tau decays from QCD jet background in simulated LHC (Large Hadron Collider) datasets.

## Dataset

- **Source**: Simulated using MAD-GRAPH → PYTHIA → DELPHES (CMS detector config).
- **Events**: Each entry represents either a Higgs decay (signal) or QCD jet (background).
- **Preprocessing**: Jets are reconstructed with the anti-kT algorithm (R = 0.4), and selected with \( p_T > 50 \text{GeV} \) and \( |\eta| < 2.5 \).
- **Ratio**: 70% background, 30% signal.

## Models and Methods

### 1. Boosted Decision Trees (BDT)
- **Features**: Track multiplicity, hadronic energy fraction, substructure features (N-subjettiness, energy correlation).
- **Tool**: XGBoost with depth 5 and 200 estimators.

### 2. Neural Networks (NN)
- **Architecture**: 3-layer MLP with sizes [256, 128, 64], BatchNorm, Dropout (0.3).
- **Input**: Same features as BDT.

### 3. Ensemble CNN + NN
- **CNN**: Lightweight CNN on 64×64 jet images (η-φ plane), centered and normalized.
- **Combined**: CNN image outputs concatenated with jet-level features and passed to NN.

### 4. ParticleNet (PartNet)
- **Approach**: Graph-based; represents jets as unordered particle sets.
- **Model**: EdgeConv-based GNN using particle-level features.
- **Optimizer**: AdamW with 1-cycle LR scheduler.

### 5. FROCC (Fast Random Projection One-Class Classification)
- **Type**: Anomaly detection
- **Parameters**: 2000 projections, epsilon = 1e-4
- **Use Case**: Train on one class to detect outliers.

### 6. POND (PrOjected Neural Density Estimation)
- **Description**: Learns projection vectors and compares density estimates against known class pivots.
- **Parameters**: vectors = 5, kernels = 2000, RBF scoring.

## Results

| Model        | Accuracy | ROC-AUC | F1-score |
|--------------|----------|---------|----------|
| BDT          | 96.10    | 0.9424  | 0.9180   |
| NN           | 96.05    | 0.9802  | 0.9416   |
| CNN + NN     | 96.15    | 0.9322  | 0.9434   |
| FROCC        | 91.43    | 0.9149  | 0.9095   |
| FROCC + NN   | 96.86    | 0.9609  | 0.9572   |
| POND         | 91.47    | 0.9006  | 91.32    |
| PartNet      | **97.47**| 0.9792  | **0.9593**|

**Note**: PartNet uses particle-level inputs; others use jet-level features.

## Future Work

We aim to evaluate the **Particle Transformer (ParT)** model, which has shown promise in literature but hasn't yet been tested on tau datasets.

## References

- Chakraborty, Iyer, and Roy, _"A Framework for Finding Anomalous Objects at the LHC"_ (arXiv:1707.07084)
- Oliveira et al., _"Jet-Images – Deep Learning Edition"_ (arXiv:1511.05190v3)
- Huilin Qu et al., _"Jet Tagging via Particle Clouds"_ (arXiv:1902.08570v3)
- Arindam Bhattacharya et al., _"FROCC: Fast Random projection-based One-Class Classification"_ (arXiv:2011.14317)
- Harsh Pandey et al., _"Stability-Based Data-Dependent Generalization Bounds for Classification: Theory and Application to a Novel Method"_

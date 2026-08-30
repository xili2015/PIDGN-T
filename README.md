# PIDGN-T

## Explainable Temporal Graph Learning for Multimodal Neurogenetic Data Integration in Early Parkinson’s Disease Prediction


This repository provides the official implementation of **PIDGN-T**, an explainable temporal graph learning framework designed for early Parkinson’s disease (PD) prediction using multimodal neurogenetic data integration.

PIDGN-T integrates heterogeneous biomedical modalities, including:

- Genomic single nucleotide polymorphism (SNP) profiles
- Structural magnetic resonance imaging (sMRI)
- Longitudinal clinical and behavioral observations

through biologically informed graph construction, cross-modal graph attention fusion (CM-GAF), and temporal graph neural network (TGNN)-based progression modeling.

The framework further incorporates multi-level explainability mechanisms, including SHAP-GNN, Grad-CAM, and temporal saliency analysis, to identify influential genetic, neuroanatomical, and longitudinal biomarkers.


---

# Repository Overview

The repository contains:

- Data preprocessing pipelines
- Modality-specific feature encoders
- Graph construction modules
- Cross-modal graph attention fusion
- Temporal graph neural network implementation
- Explainability modules
- Baseline model implementations
- Training and evaluation scripts
- Statistical analysis and robustness evaluation tools


---

# Framework Components

The PIDGN-T framework consists of the following main modules:

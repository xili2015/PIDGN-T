# Dataset and Data Preparation

This directory contains the data preparation pipeline used for the PIDGN-T framework.

The proposed framework was evaluated on the Parkinson’s Progression Markers Initiative (PPMI) dataset, which provides longitudinal multimodal information including:

- Genomic single nucleotide polymorphism (SNP) profiles
- Structural magnetic resonance imaging (sMRI)
- Clinical assessments
- Behavioral measurements


## Dataset Access

The PPMI dataset is not redistributed in this repository due to data usage restrictions and participant privacy policies.

Researchers can request access through:

https://www.ppmi-info.org/


## Directory Organization

## Subject-Level Data Splitting

To prevent information leakage in longitudinal Parkinson’s disease prediction,
all data splits were performed at the subject level.

All visits belonging to the same participant were assigned exclusively to one
split (training, validation, or testing).

No longitudinal observation from the test subjects was used during model
optimization or hyperparameter selection.

## MRI Processing Pipeline

The MRI preprocessing workflow consists of:

1. Intensity normalization and spatial standardization.
2. Anatomical brain parcellation using predefined atlases.
3. Deep feature extraction using a 3D residual convolutional encoder.

The extracted regional embeddings are used as node attributes for the Brain Connectome Graph in PIDGN-T.


## Clinical and Behavioral Processing

The clinical preprocessing pipeline performs:

1. Subject-level multimodal synchronization.
2. Longitudinal visit alignment.
3. Missing value handling and feature normalization.
4. Temporal sequence generation for recurrent modeling.

The generated sequences are provided to the Clinical Encoder and subsequently integrated into the Temporal Graph Neural Network (TGNN).

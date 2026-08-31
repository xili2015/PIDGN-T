"""
============================================================
PIDGN-T
Clinical Feature Scaling Module

Purpose:
    Normalize heterogeneous clinical and behavioral
    variables before temporal representation learning.

Supported features:

- UPDRS scores
- Cognitive scores
- Tremor measurements
- Gait characteristics
- Behavioral signals

============================================================
"""


import pandas as pd
import numpy as np



def z_score_scaling(
        data,
        columns
):

    """
    Apply z-score normalization.

    Formula:

        x' = (x - mean) / std

    """

    scaled_data = data.copy()


    for col in columns:

        mean = data[col].mean()

        std = data[col].std()


        scaled_data[col] = (
            data[col] - mean
        ) / (
            std + 1e-8
        )


    return scaled_data



def min_max_scaling(
        data,
        columns
):

    """
    Min-max normalization.

    Formula:

        x'=(x-min)/(max-min)

    """

    scaled_data = data.copy()


    for col in columns:

        minimum = data[col].min()

        maximum = data[col].max()


        scaled_data[col] = (
            data[col]-minimum
        ) / (
            maximum-minimum+1e-8
        )


    return scaled_data



def handle_missing_features(
        data
):

    """
    Replace missing clinical values
    using median imputation.
    """

    numeric_columns = (
        data.select_dtypes(
            include=np.number
        )
        .columns
    )


    data[numeric_columns] = (
        data[numeric_columns]
        .fillna(
            data[numeric_columns]
            .median()
        )
    )


    return data



def clinical_scaling_pipeline(
        data,
        feature_columns
):

    """
    Complete clinical preprocessing pipeline.
    """

    data = handle_missing_features(
        data
    )


    data = z_score_scaling(
        data,
        feature_columns
    )


    return data



if __name__ == "__main__":

    print(
        "Clinical feature scaling module ready."
    )

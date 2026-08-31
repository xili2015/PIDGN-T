"""
============================================================
PIDGN-T
Temporal Sequence Construction Module

Purpose:
    Convert longitudinal clinical observations
    into temporal sequences for recurrent learning.

Input:
    Aligned clinical records

Output:
    Patient-level temporal sequences

============================================================
"""


import numpy as np
import pandas as pd



def generate_patient_sequences(
        data,
        feature_columns
):

    """
    Generate temporal sequences for each patient.

    Each patient is represented as:

    X = {x1, x2, ..., xt}

    where each xt represents a clinical
    observation at visit t.
    """

    sequences = []

    labels = []



    for subject_id, patient_data in data.groupby(
            "subject_id"
    ):

        patient_data = patient_data.sort_values(
            "visit_time"
        )


        sequence = (
            patient_data[feature_columns]
            .values
        )


        sequences.append(
            sequence
        )


        labels.append(
            patient_data["label"]
            .iloc[0]
        )


    return sequences, labels



def pad_sequences(
        sequences,
        max_length=None
):

    """
    Pad variable-length longitudinal sequences.

    Required because different patients
    have different follow-up durations.
    """


    if max_length is None:

        max_length = max(
            len(seq)
            for seq in sequences
        )


    feature_dim = sequences[0].shape[1]


    padded = np.zeros(
        (
            len(sequences),
            max_length,
            feature_dim
        )
    )



    for i, seq in enumerate(sequences):

        length = min(
            len(seq),
            max_length
        )


        padded[
            i,
            :length,
            :
        ] = seq[
            :length
        ]


    return padded



def build_temporal_input(
        clinical_data,
        features
):

    """
    Complete temporal sequence pipeline.
    """

    sequences, labels = (
        generate_patient_sequences(
            clinical_data,
            features
        )
    )


    sequences = pad_sequences(
        sequences
    )


    return sequences, labels



if __name__ == "__main__":

    print(
        "Temporal sequence module ready."
    )

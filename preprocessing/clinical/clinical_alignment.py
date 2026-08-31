"""
============================================================
PIDGN-T
Clinical Alignment Module

Purpose:
    Align longitudinal clinical and behavioral observations
    across multiple patient visits.

Input:
    Raw clinical records

Output:
    Subject-level aligned longitudinal data

============================================================
"""


import pandas as pd



def load_clinical_data(path):

    """
    Load clinical and behavioral records.

    Required columns:

    subject_id
    visit_time
    clinical variables

    """

    data = pd.read_csv(path)

    return data



def sort_longitudinal_visits(data):

    """
    Sort patient records according to
    chronological visit order.
    """

    aligned_data = data.sort_values(
        [
            "subject_id",
            "visit_time"
        ]
    )

    return aligned_data



def remove_inconsistent_visits(
        data,
        minimum_visits=2
):

    """
    Remove subjects with insufficient
    longitudinal observations.

    Patients with fewer than the required
    number of visits are excluded.
    """

    visit_count = (
        data.groupby(
            "subject_id"
        )
        .size()
    )


    valid_subjects = visit_count[
        visit_count >= minimum_visits
    ].index


    filtered_data = data[
        data["subject_id"]
        .isin(valid_subjects)
    ]


    return filtered_data



def create_modality_alignment(
        clinical_data,
        genomic_ids,
        imaging_ids
):

    """
    Perform subject-level alignment among:

    - genomic data
    - MRI data
    - clinical observations

    Only subjects available in all modalities
    are retained.
    """

    common_subjects = (
        set(clinical_data["subject_id"])
        &
        set(genomic_ids)
        &
        set(imaging_ids)
    )


    aligned = clinical_data[
        clinical_data["subject_id"]
        .isin(common_subjects)
    ]


    return aligned



def clinical_alignment_pipeline(
        path
):

    """
    Complete clinical alignment pipeline.
    """

    data = load_clinical_data(
        path
    )


    data = sort_longitudinal_visits(
        data
    )


    data = remove_inconsistent_visits(
        data
    )


    return data



if __name__ == "__main__":

    print(
        "Clinical alignment module ready."
    )

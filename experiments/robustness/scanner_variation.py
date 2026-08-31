"""
============================================================
PIDGN-T Robustness Experiment

Cross-Scanner Variation Simulation

Purpose:
    Simulate MRI scanner variability.

Scenario:
    Intensity scaling and acquisition shift.

Corresponds to:
    Section 4.5.4

============================================================
"""


import torch





def simulate_scanner_shift(
        mri_features,
        intensity_factor=0.95
):

    """
    Simulate scanner-dependent variation.


    Parameters
    ----------
    mri_features:

        MRI representation


    intensity_factor:

        Scanner intensity scaling factor


    Returns
    -------
    Modified MRI features

    """


    shifted_features = (

        mri_features *

        intensity_factor

    )


    return shifted_features





def generate_scanner_variations(
        features,
        factors=[0.90,0.95,1.05,1.10]
):


    """

    Create different scanner conditions.

    """


    variations={}


    for factor in factors:


        variations[factor] = simulate_scanner_shift(

            features,

            factor

        )


    return variations





if __name__ == "__main__":


    print(

        "Scanner variation robustness module ready."

    )

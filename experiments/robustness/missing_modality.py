"""
============================================================
PIDGN-T Robustness Experiment

Missing Modality Simulation

Purpose:
    Evaluate robustness when one modality
    is partially unavailable.

Scenario:
    Random masking of MRI features

Default:
    Missing rate = 10%

============================================================
"""


import torch





def apply_missing_modality(
        modality_features,
        missing_rate=0.10
):

    """
    Randomly mask modality features.


    Parameters
    ----------
    modality_features:

        MRI latent representation


    missing_rate:

        Percentage of removed information


    Returns
    -------
    Masked modality representation

    """



    mask = (

        torch.rand_like(

            modality_features

        )

        >

        missing_rate

    )



    masked_features = (

        modality_features * mask

    )



    return masked_features





def generate_missing_scenarios(
        features,
        rates=[0.05,0.10,0.20]
):


    """

    Generate multiple missing modality cases.

    """


    scenarios={}


    for rate in rates:


        scenarios[rate] = apply_missing_modality(

            features,

            rate

        )


    return scenarios





if __name__ == "__main__":


    print(

        "Missing modality robustness module ready."

    )

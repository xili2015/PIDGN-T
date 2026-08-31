"""
============================================================
PIDGN-T Robustness Experiment

Gaussian Noise Perturbation

Purpose:
    Evaluate model stability when genomic
    embeddings are affected by noise.

Scenario:
    Add Gaussian noise with sigma=0.05

Corresponds to:
    Section 4.5.4

============================================================
"""


import torch



def add_gaussian_noise(
        embeddings,
        sigma=0.05
):

    """
    Add Gaussian noise.

    x' = x + N(0,sigma)

    Parameters
    ----------
    embeddings:
        Original latent representation

    sigma:
        Noise intensity

    Returns
    -------
    Perturbed embeddings
    """


    noise = torch.normal(

        mean=0.0,

        std=sigma,

        size=embeddings.shape,

        device=embeddings.device

    )


    noisy_embeddings = (

        embeddings + noise

    )


    return noisy_embeddings





def evaluate_noise_levels(
        embeddings,
        noise_levels
):

    """
    Generate multiple noisy versions
    for robustness evaluation.
    """


    results={}


    for sigma in noise_levels:


        results[sigma] = add_gaussian_noise(

            embeddings,

            sigma

        )


    return results





if __name__ == "__main__":


    print(

        "Gaussian noise robustness module ready."

    )

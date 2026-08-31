"""
============================================================
PIDGN-T
Genomic Embedding Generation Module

Purpose:
    Generate latent genomic representations using
    an autoencoder-based feature learning strategy.

Input:
    Normalized SNP matrix

Output:
    Low-dimensional genomic embeddings

============================================================
"""


import torch
import torch.nn as nn



class GenomicAutoencoder(
        nn.Module
):

    """
    Denoising Autoencoder for SNP representation learning.
    """

    def __init__(
            self,
            input_dim,
            latent_dim=128
    ):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),


            nn.Linear(
                512,
                latent_dim
            )

        )


        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                512
            ),

            nn.ReLU(),


            nn.Linear(
                512,
                input_dim
            )

        )



    def forward(
            self,
            x
    ):

        embedding = self.encoder(x)

        reconstruction = self.decoder(
            embedding
        )


        return reconstruction, embedding



def generate_genomic_embedding(
        snp_matrix,
        latent_dimension=128
):

    """
    Extract latent genomic representation.

    Parameters
    ----------
    snp_matrix :
        Normalized SNP tensor


    Returns
    -------
    genomic embedding
    """

    input_dimension = (
        snp_matrix.shape[1]
    )


    model = GenomicAutoencoder(
        input_dimension,
        latent_dimension
    )


    with torch.no_grad():

        _, embedding = model(
            snp_matrix
        )


    return embedding



if __name__ == "__main__":

    print(
        "Genomic embedding module ready."
    )

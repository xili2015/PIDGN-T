"""
============================================================
PIDGN-T
Genomic Autoencoder Encoder

Purpose:
    Learn compact latent representations from
    high-dimensional SNP genomic profiles.

Input:
    Normalized SNP feature vectors

Output:
    Genomic latent embeddings

Corresponds to:
    Section 3.2.1 Genomic Encoder

============================================================
"""


import torch
import torch.nn as nn



class GenomicAutoencoder(nn.Module):

    """
    Denoising Autoencoder for genomic SNP representation.

    The encoder compresses high-dimensional SNP vectors
    into biologically meaningful latent embeddings.
    """

    def __init__(
            self,
            input_dim,
            latent_dim=128,
            dropout=0.3
    ):

        super(
            GenomicAutoencoder,
            self
        ).__init__()



        # Encoder network

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),


            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),


            nn.Linear(
                256,
                latent_dim
            )

        )


        # Decoder network

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                256
            ),

            nn.ReLU(),


            nn.Linear(
                256,
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

        """
        Forward propagation.

        Returns:
            reconstruction
            latent genomic embedding
        """

        embedding = self.encoder(
            x
        )


        reconstruction = self.decoder(
            embedding
        )


        return reconstruction, embedding



def extract_genomic_embedding(
        model,
        snp_features
):

    """
    Generate genomic latent representation.

    Parameters
    ----------
    model:
        Trained genomic autoencoder

    snp_features:
        Normalized SNP tensor


    Returns
    -------
    latent genomic embedding
    """

    model.eval()


    with torch.no_grad():

        _, embedding = model(
            snp_features
        )


    return embedding



if __name__ == "__main__":

    print(
        "Genomic Autoencoder module ready."
    )

"""
============================================================
PIDGN-T
MRI Feature Extraction Module

Purpose:
    Extract deep spatial representations from
    structural MRI volumes.

Architecture:
    3D Residual CNN Encoder

Output:
    MRI latent embeddings

============================================================
"""


import torch
import torch.nn as nn



class ResNet3DEncoder(
        nn.Module
):

    """
    Lightweight 3D-ResNet encoder.

    Generates MRI latent representations
    for graph-based learning.
    """



    def __init__(
            self,
            input_channels=1,
            feature_dimension=256
    ):

        super().__init__()



        self.encoder = nn.Sequential(

            nn.Conv3d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(
                32
            ),

            nn.ReLU(),


            nn.MaxPool3d(
                2
            ),


            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(
                64
            ),

            nn.ReLU(),


            nn.AdaptiveAvgPool3d(
                1
            )

        )


        self.fc = nn.Linear(

            64,

            feature_dimension

        )



    def forward(
            self,
            x
    ):

        x = self.encoder(x)

        x = torch.flatten(
            x,
            1
        )

        embedding = self.fc(
            x
        )


        return embedding



def extract_mri_embedding(
        mri_tensor
):

    """
    Generate MRI latent representation.

    Parameters
    ----------
    mri_tensor :
        Normalized MRI volume


    Returns
    -------
    MRI embedding vector
    """

    model = ResNet3DEncoder()


    model.eval()


    with torch.no_grad():

        embedding = model(
            mri_tensor
        )


    return embedding



if __name__ == "__main__":

    print(
        "MRI feature extraction module ready."
    )

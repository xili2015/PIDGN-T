"""
============================================================
PIDGN-T
3D ResNet MRI Encoder

Purpose:
    Extract spatial representations from
    structural MRI volumes.

Input:
    Normalized T1-weighted MRI volumes

Output:
    MRI latent embeddings

Corresponds to:
    Section 3.2.2 Structural MRI Encoder

============================================================
"""


import torch
import torch.nn as nn



class ResidualBlock3D(
        nn.Module
):

    """
    Basic residual block for 3D MRI feature learning.
    """

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()


        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )


        self.bn1 = nn.BatchNorm3d(
            out_channels
        )


        self.relu = nn.ReLU()


        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )


        self.bn2 = nn.BatchNorm3d(
            out_channels
        )


        self.shortcut = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1
        )



    def forward(
            self,
            x
    ):


        identity = self.shortcut(
            x
        )


        out = self.conv1(
            x
        )

        out = self.bn1(
            out
        )

        out = self.relu(
            out
        )


        out = self.conv2(
            out
        )

        out = self.bn2(
            out
        )


        out += identity


        return self.relu(
            out
        )




class ResNet3DEncoder(
        nn.Module
):


    """
    3D Residual CNN encoder for MRI feature extraction.
    """

    def __init__(
            self,
            input_channels=1,
            embedding_dim=256
    ):

        super().__init__()


        self.features = nn.Sequential(

            ResidualBlock3D(
                input_channels,
                32
            ),

            nn.MaxPool3d(
                2
            ),


            ResidualBlock3D(
                32,
                64
            ),

            nn.MaxPool3d(
                2
            ),


            ResidualBlock3D(
                64,
                128
            ),


            nn.AdaptiveAvgPool3d(
                1
            )

        )


        self.embedding = nn.Linear(

            128,

            embedding_dim

        )



    def forward(
            self,
            x
    ):

        """

        Extract MRI embedding.

        """

        x = self.features(
            x
        )


        x = torch.flatten(
            x,
            1
        )


        embedding = self.embedding(
            x
        )


        return embedding



if __name__ == "__main__":

    print(
        "3D ResNet MRI encoder ready."
    )

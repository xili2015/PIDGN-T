"""
============================================================
Baseline: PIDGN

Original Multimodal Graph Learning Framework

Purpose:
    Baseline implementation of PIDGN without
    temporal graph reasoning.

Components:
    - Genomic encoder
    - MRI encoder
    - Graph attention fusion
    - Classifier

============================================================
"""


import torch
import torch.nn as nn



class PIDGN(nn.Module):


    def __init__(
            self,
            genomic_dim,
            imaging_dim,
            hidden_dim=128,
            num_classes=2
    ):

        super().__init__()



        self.genomic_projection = nn.Linear(

            genomic_dim,

            hidden_dim

        )


        self.imaging_projection = nn.Linear(

            imaging_dim,

            hidden_dim

        )


        self.attention = nn.MultiheadAttention(

            hidden_dim,

            num_heads=4,

            batch_first=True

        )


        self.classifier = nn.Sequential(

            nn.Linear(

                hidden_dim,

                64

            ),

            nn.ReLU(),


            nn.Linear(

                64,

                num_classes

            )

        )



    def forward(

            self,

            genomic_features,

            imaging_features

    ):


        g = self.genomic_projection(
            genomic_features
        )


        i = self.imaging_projection(
            imaging_features
        )


        x = torch.stack(
            [
                g,
                i
            ],
            dim=1
        )


        fused, attention = self.attention(
            x,
            x,
            x
        )


        representation = fused.mean(
            dim=1
        )


        output = self.classifier(
            representation
        )


        return output

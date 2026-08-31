"""
============================================================
Baseline: Multimodal Transformer (MMT)

Purpose:
    Transformer-based multimodal fusion.

============================================================
"""


import torch.nn as nn



class MMT(nn.Module):


    def __init__(
            self,
            feature_dim,
            classes=2
    ):

        super().__init__()


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=feature_dim,

            nhead=4,

            batch_first=True

        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=2

        )


        self.fc = nn.Linear(

            feature_dim,

            classes

        )



    def forward(
            self,
            x
    ):


        x=self.transformer(
            x
        )


        return self.fc(
            x.mean(dim=1)
        )

"""
============================================================
Baseline: DySAT

Dynamic Graph Self-Attention Network

============================================================
"""


import torch
import torch.nn as nn



class DySAT(nn.Module):


    def __init__(
            self,
            input_dim,
            hidden_dim,
            classes=2
    ):

        super().__init__()


        self.encoder = nn.Linear(

            input_dim,

            hidden_dim

        )


        self.temporal_attention = nn.MultiheadAttention(

            hidden_dim,

            num_heads=4,

            batch_first=True

        )


        self.fc = nn.Linear(

            hidden_dim,

            classes

        )



    def forward(
            self,
            graph_sequence
    ):


        x=self.encoder(
            graph_sequence
        )


        x,_=self.temporal_attention(

            x,

            x,

            x

        )


        return self.fc(
            x.mean(dim=1)
        )

"""
============================================================
Baseline: EvolveGCN

Purpose:
    Dynamic evolution of GCN parameters
    over graph snapshots.

============================================================
"""


import torch
import torch.nn as nn



class EvolveGCN(nn.Module):


    def __init__(
            self,
            input_dim,
            hidden_dim,
            classes=2
    ):

        super().__init__()


        self.gcn = nn.Linear(

            input_dim,

            hidden_dim

        )


        self.gru = nn.GRU(

            hidden_dim,

            hidden_dim,

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


        x=self.gcn(
            graph_sequence
        )


        x,_=self.gru(
            x
        )


        return self.fc(
            x[:,-1]
        )

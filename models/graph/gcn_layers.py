"""
============================================================
PIDGN-T
Graph Convolution Layers

Purpose:
    Perform graph message passing using
    normalized adjacency matrices.

Corresponds to:
    Section 3.3 Graph Representation Learning

============================================================
"""


import torch
import torch.nn as nn



class GraphConvolution(
        nn.Module
):


    def __init__(
            self,
            input_dim,
            output_dim
    ):

        super().__init__()


        self.linear = nn.Linear(
            input_dim,
            output_dim
        )



    def forward(
            self,
            x,
            adjacency
    ):

        """
        Graph convolution operation:

        H'=σ(AHW)

        """

        aggregated = torch.matmul(
            adjacency,
            x
        )


        output = self.linear(
            aggregated
        )


        return torch.relu(
            output
        )




class GCNEncoder(
        nn.Module
):


    """
    Multi-layer GCN encoder.
    """


    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim
    ):

        super().__init__()


        self.gcn1 = GraphConvolution(
            input_dim,
            hidden_dim
        )


        self.gcn2 = GraphConvolution(
            hidden_dim,
            output_dim
        )



    def forward(
            self,
            x,
            adjacency
    ):


        x = self.gcn1(
            x,
            adjacency
        )


        x = self.gcn2(
            x,
            adjacency
        )


        return x



if __name__ == "__main__":

    print(
        "GCN layers module ready."
    )

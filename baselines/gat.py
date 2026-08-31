"""
============================================================
Baseline: Graph Attention Network (GAT)

Purpose:
    Graph-based representation learning using
    attention-based message passing.

============================================================
"""


import torch
import torch.nn as nn



class GATLayer(nn.Module):


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


        self.attention = nn.Linear(

            output_dim * 2,

            1

        )



    def forward(

            self,

            x,

            adjacency

    ):


        h = self.linear(
            x
        )


        n = h.size(0)


        h1 = h.repeat(
            n,
            1
        )


        h2 = h.repeat_interleave(
            n,
            dim=0
        )


        attention_scores = torch.sigmoid(

            self.attention(

                torch.cat(
                    [
                        h1,
                        h2
                    ],
                    dim=1
                )

            )

        )


        attention_scores = attention_scores.view(
            n,
            n
        )


        attention_scores *= adjacency


        output = torch.matmul(

            attention_scores,

            h

        )


        return torch.relu(
            output
        )



class GAT(nn.Module):


    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_classes
    ):

        super().__init__()


        self.gat1 = GATLayer(
            input_dim,
            hidden_dim
        )


        self.classifier = nn.Linear(

            hidden_dim,

            num_classes

        )


    def forward(
            self,
            x,
            adjacency
    ):

        x = self.gat1(
            x,
            adjacency
        )


        return self.classifier(
            x.mean(dim=0)
        )

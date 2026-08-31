"""
============================================================
PIDGN-T
Temporal Graph Neural Network (TGNN)

Purpose:
    Model longitudinal disease progression by integrating
    graph structural dependencies and temporal dynamics.

Architecture:
    Graph Convolution + GRU + Temporal Attention

Input:
    Multimodal graph embeddings from CM-GAF

Output:
    Temporal patient representation

Corresponds to:
    Section 3.5 Temporal Graph Neural Network

============================================================
"""


import torch
import torch.nn as nn



# ============================================================
# Graph Convolution Unit
# ============================================================


class GraphConvolution(nn.Module):

    """
    Basic graph message passing layer.

    H(t+1)=σ(AHW)
    """

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
            node_features,
            adjacency
    ):

        """

        Parameters
        ----------
        node_features:
            Node embeddings

        adjacency:
            Graph adjacency matrix

        """

        aggregated = torch.matmul(
            adjacency,
            node_features
        )


        output = self.linear(
            aggregated
        )


        return torch.relu(
            output
        )



# ============================================================
# TGNN Module
# ============================================================


class TGNN(nn.Module):

    """
    Temporal Graph Neural Network.

    Components:

    1. Spatial graph learning using GCN
    2. Temporal dependency learning using GRU
    3. Longitudinal representation generation

    """



    def __init__(
            self,
            input_dim,
            hidden_dim=256,
            graph_dim=128,
            num_layers=1
    ):

        super().__init__()



        self.gcn = GraphConvolution(

            input_dim,

            graph_dim

        )



        self.gru = nn.GRU(

            input_size=graph_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True

        )



        self.temporal_projection = nn.Linear(

            hidden_dim,

            hidden_dim

        )



    def forward(

            self,

            graph_sequences,

            adjacency

    ):

        """
        Forward propagation.

        Parameters
        ----------
        graph_sequences:

            Tensor:

            Batch x Time x Nodes x Features


        adjacency:

            Graph structure


        Returns
        -------
        Temporal hidden representation

        """


        batch_size, time_steps, nodes, features = (
            graph_sequences.shape
        )


        temporal_embeddings = []



        for t in range(time_steps):


            node_features = graph_sequences[:, t]


            spatial_embedding = self.gcn(

                node_features,

                adjacency

            )


            # Node aggregation

            graph_embedding = torch.mean(

                spatial_embedding,

                dim=1

            )


            temporal_embeddings.append(

                graph_embedding

            )



        temporal_sequence = torch.stack(

            temporal_embeddings,

            dim=1

        )


        outputs, hidden = self.gru(

            temporal_sequence

        )


        final_state = self.temporal_projection(

            hidden[-1]

        )


        return {

            "temporal_embedding":

                final_state,


            "sequence_features":

                outputs

        }



if __name__ == "__main__":

    print(
        "TGNN module ready."
    )

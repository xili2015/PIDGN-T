"""
============================================================
PIDGN-T
Cross-Modal Graph Attention Fusion (CM-GAF)

Purpose:
    Learn biologically meaningful interactions between
    heterogeneous graph modalities.

Input:
    1. Gene Interaction Graph embeddings
    2. Brain Connectome Graph embeddings

Output:
    Unified multimodal graph representation

Main Components:
    - Cross-modal attention
    - Multi-head graph attention
    - Bidirectional consistency learning

Corresponds to:
    Section 3.4 Cross-Modal Graph Attention Fusion

============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# Cross Modal Attention Layer
# ============================================================


class CrossModalAttention(nn.Module):

    """
    Computes attention between two graph modalities.

    Example:

    Gene nodes ---> Brain nodes

    The attention score indicates the contribution
    of each gene representation to brain regions.
    """



    def __init__(
            self,
            dim,
            attention_dim
    ):

        super().__init__()



        self.query_projection = nn.Linear(
            dim,
            attention_dim
        )


        self.key_projection = nn.Linear(
            dim,
            attention_dim
        )


        self.value_projection = nn.Linear(
            dim,
            dim
        )


        self.scale = (
            attention_dim ** -0.5
        )



    def forward(
            self,
            source_features,
            target_features
    ):

        """
        Parameters
        ----------
        source_features:
            Source modality node embeddings

        target_features:
            Target modality node embeddings


        Returns
        -------
        fused representation
        attention weights
        """



        Q = self.query_projection(
            target_features
        )


        K = self.key_projection(
            source_features
        )


        V = self.value_projection(
            source_features
        )



        attention_scores = torch.matmul(

            Q,

            K.transpose(
                -2,
                -1
            )

        ) * self.scale



        attention_weights = F.softmax(

            attention_scores,

            dim=-1

        )



        attended_features = torch.matmul(

            attention_weights,

            V

        )


        return (
            attended_features,
            attention_weights
        )





# ============================================================
# Multi-head CM-GAF Module
# ============================================================


class CMGAF(nn.Module):


    """
    Cross-Modal Graph Attention Fusion module.

    It integrates:

    - Gene Interaction Graph
    - Brain Connectome Graph


    using multi-head attention.
    """



    def __init__(
            self,
            gene_dim,
            brain_dim,
            hidden_dim=128,
            heads=8,
            dropout=0.2
    ):

        super().__init__()



        self.heads = heads



        self.gene_projection = nn.Linear(

            gene_dim,

            hidden_dim

        )


        self.brain_projection = nn.Linear(

            brain_dim,

            hidden_dim

        )



        self.attention_heads = nn.ModuleList(

            [

                CrossModalAttention(

                    hidden_dim,

                    hidden_dim

                )

                for _ in range(heads)

            ]

        )



        self.output_projection = nn.Linear(

            hidden_dim * heads,

            hidden_dim

        )



        self.dropout = nn.Dropout(

            dropout

        )





    def forward(

            self,

            gene_embedding,

            brain_embedding

    ):

        """
        Perform bidirectional cross-modal fusion.

        Parameters
        ----------
        gene_embedding:

            Gene graph node features


        brain_embedding:

            Brain graph node features



        Returns
        -------
        fused representation
        attention maps

        """



        gene_features = self.gene_projection(

            gene_embedding

        )


        brain_features = self.brain_projection(

            brain_embedding

        )



        head_outputs = []

        attention_maps = []



        # Gene -> Brain attention

        for attention_layer in self.attention_heads:


            fused_gene_to_brain, weights = (

                attention_layer(

                    gene_features,

                    brain_features

                )

            )


            head_outputs.append(

                fused_gene_to_brain

            )


            attention_maps.append(

                weights

            )



        fused = torch.cat(

            head_outputs,

            dim=-1

        )



        fused = self.output_projection(

            fused

        )



        fused = self.dropout(

            fused

        )



        return (

            fused,

            attention_maps

        )





# ============================================================
# Bidirectional Consistency Loss
# ============================================================


class AlignmentLoss(nn.Module):

    """
    Enforces consistency between
    gene and brain representations.

    Corresponds to:

    L_align

    in PIDGN-T objective function.
    """



    def __init__(self):

        super().__init__()



    def forward(

            self,

            gene_embedding,

            brain_embedding

    ):


        loss = torch.mean(

            (

                gene_embedding -
                brain_embedding

            ) ** 2

        )


        return loss





# ============================================================
# Complete CM-GAF Pipeline
# ============================================================


class CrossModalGraphAttentionFusion(nn.Module):


    """
    Complete CM-GAF framework.

    Pipeline:

    Gene Graph
          |
          |
          v
    Cross Modal Attention
          |
          |
    Brain Connectome Graph


          |
          v

    Unified Multimodal Representation

    """



    def __init__(

            self,

            gene_dim,

            brain_dim,

            hidden_dim=128,

            heads=8

    ):


        super().__init__()



        self.cm_gaf = CMGAF(

            gene_dim,

            brain_dim,

            hidden_dim,

            heads

        )


        self.alignment_loss = AlignmentLoss()



    def forward(

            self,

            gene_graph,

            brain_graph

    ):


        fused_features, attention = (

            self.cm_gaf(

                gene_graph,

                brain_graph

            )

        )



        return {

            "fusion_embedding":

                fused_features,


            "attention_weights":

                attention

        }



if __name__ == "__main__":


    print(

        "CM-GAF fusion module ready."

    )

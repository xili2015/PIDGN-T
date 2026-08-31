"""
============================================================
PIDGN-T
Gene Interaction Graph Module

Purpose:
    Construct biologically informed gene interaction graph
    from genomic embeddings.

Graph:
    Nodes  : Genes/SNP-derived representations
    Edges  : Biological relationships

Corresponds to:
    Section 3.3.1 Gene Interaction Graph

============================================================
"""


import torch
import torch.nn as nn



class GeneInteractionGraph:


    def __init__(
            self,
            threshold=0.5
    ):

        self.threshold = threshold



    def compute_similarity(
            self,
            gene_features
    ):

        """
        Compute pairwise similarity between
        gene embeddings.

        Cosine similarity is used to
        estimate functional relationships.
        """

        normalized = torch.nn.functional.normalize(
            gene_features,
            p=2,
            dim=1
        )


        similarity = torch.mm(
            normalized,
            normalized.T
        )


        return similarity



    def build_adjacency(
            self,
            gene_features
    ):

        """
        Construct adjacency matrix.

        Edge exists if similarity exceeds
        predefined threshold.
        """

        similarity = self.compute_similarity(
            gene_features
        )


        adjacency = (
            similarity >
            self.threshold
        ).float()


        # Add self connections

        identity = torch.eye(
            adjacency.size(0)
        )


        adjacency = adjacency + identity


        return adjacency



    def forward(
            self,
            gene_features
    ):

        """
        Generate gene graph.
        """

        adjacency = self.build_adjacency(
            gene_features
        )


        return {

            "node_features":
                gene_features,

            "adjacency":
                adjacency

        }



if __name__ == "__main__":

    print(
        "Gene Interaction Graph module ready."
    )

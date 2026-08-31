"""
============================================================
PIDGN-T
Brain Connectome Graph Module

Purpose:
    Construct subject-specific brain graph
    from MRI-derived regional embeddings.

Nodes:
    Brain anatomical regions

Edges:
    Structural similarity/connectivity

Corresponds to:
    Section 3.3.2 Cerebral Connectome Graph

============================================================
"""


import torch



class BrainConnectomeGraph:


    def __init__(
            self,
            sigma=1.0
    ):

        self.sigma = sigma



    def gaussian_similarity(
            self,
            region_features
    ):

        """
        Calculate Gaussian kernel similarity.

        A(i,j)=exp(-||xi-xj||²/2σ²)

        """

        distance = torch.cdist(
            region_features,
            region_features
        )


        adjacency = torch.exp(
            -distance ** 2 /
            (2 * self.sigma ** 2)
        )


        return adjacency



    def normalize_adjacency(
            self,
            adjacency
    ):

        """
        Symmetric adjacency normalization.
        """

        degree = torch.sum(
            adjacency,
            dim=1
        )


        degree_inv = torch.pow(
            degree + 1e-8,
            -0.5
        )


        D = torch.diag(
            degree_inv
        )


        normalized = torch.mm(
            torch.mm(
                D,
                adjacency
            ),
            D
        )


        return normalized



    def forward(
            self,
            brain_features
    ):

        adjacency = self.gaussian_similarity(
            brain_features
        )


        adjacency = self.normalize_adjacency(
            adjacency
        )


        return {

            "node_features":
                brain_features,

            "adjacency":
                adjacency

        }



if __name__ == "__main__":

    print(
        "Brain Connectome Graph module ready."
    )

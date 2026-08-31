"""
============================================================
PIDGN-T
SHAP-GNN Explainability Module

Purpose:
    Provide graph-level explanations by estimating
    contribution of nodes and graph structures.

Input:
    Graph embeddings and model predictions

Output:
    Node importance scores

Corresponds to:
    Section 3.6.1 Graph-Based Explainability
============================================================
"""


import torch
import numpy as np



class SHAPGNNExplainer:


    def __init__(
            self,
            model,
            num_samples=100
    ):

        self.model = model
        self.num_samples = num_samples



    def mask_graph_nodes(
            self,
            node_features,
            mask
    ):

        """
        Apply node masking during SHAP approximation.
        """

        masked_features = (
            node_features *
            mask.unsqueeze(-1)
        )

        return masked_features



    def approximate_shap_values(
            self,
            graph_features,
            adjacency
    ):

        """
        Approximate Shapley values using
        Monte-Carlo neighborhood perturbation.

        Returns:
            Node contribution scores
        """

        num_nodes = graph_features.size(0)


        shap_values = torch.zeros(
            num_nodes
        )


        baseline_prediction = (
            self.model(
                graph_features,
                adjacency
            )
        )



        for _ in range(
                self.num_samples
        ):

            random_mask = torch.randint(
                0,
                2,
                (num_nodes,)
            ).float()


            masked_graph = (
                self.mask_graph_nodes(
                    graph_features,
                    random_mask
                )
            )


            prediction = self.model(
                masked_graph,
                adjacency
            )


            contribution = (
                prediction -
                baseline_prediction
            )


            shap_values += (
                contribution *
                random_mask
            )



        shap_values /= self.num_samples


        return shap_values



    def explain(
            self,
            graph_features,
            adjacency
    ):

        """
        Generate graph explanation.
        """

        importance = (
            self.approximate_shap_values(
                graph_features,
                adjacency
            )
        )


        return {

            "node_importance":
                importance,


            "ranking":
                torch.argsort(
                    importance,
                    descending=True
                )

        }



if __name__ == "__main__":

    print(
        "SHAP-GNN explanation module ready."
    )

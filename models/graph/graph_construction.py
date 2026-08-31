"""
============================================================
PIDGN-T
Unified Graph Construction Pipeline

Purpose:
    Construct multimodal biological graphs
    required for CM-GAF and TGNN.

Graphs:

1. Gene Interaction Graph
2. Brain Connectome Graph

============================================================
"""


from .gene_graph import GeneInteractionGraph

from .connectome_graph import BrainConnectomeGraph



class GraphConstructionPipeline:



    def __init__(
            self
    ):


        self.gene_builder = (
            GeneInteractionGraph()
        )


        self.brain_builder = (
            BrainConnectomeGraph()
        )



    def construct_gene_graph(
            self,
            genomic_embedding
    ):

        """
        Generate biological gene graph.
        """

        return self.gene_builder.forward(
            genomic_embedding
        )



    def construct_brain_graph(
            self,
            brain_embedding
    ):

        """
        Generate brain connectome graph.
        """

        return self.brain_builder.forward(
            brain_embedding
        )



    def build_all_graphs(
            self,
            genomic_embedding,
            brain_embedding
    ):

        """
        Construct complete multimodal graph structure.
        """

        gene_graph = (
            self.construct_gene_graph(
                genomic_embedding
            )
        )


        brain_graph = (
            self.construct_brain_graph(
                brain_embedding
            )
        )


        return {

            "gene_graph":
                gene_graph,

            "brain_graph":
                brain_graph

        }



if __name__ == "__main__":

    print(
        "Graph construction pipeline ready."
    )

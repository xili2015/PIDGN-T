"""
============================================================
PIDGN-T

Explainable Temporal Graph Learning Framework
for Multimodal Neurogenetic Data Integration
in Early Parkinson's Disease Prediction


Main Architecture:

Genomic Encoder
        |
MRI Encoder
        |
Clinical Temporal Encoder
        |
        v
Biological Graph Construction
        |
        v
Cross-Modal Graph Attention Fusion (CM-GAF)
        |
        v
Temporal Graph Neural Network (TGNN)
        |
        v
Prediction + Explainability


Corresponds to:
    Complete PIDGN-T Framework

============================================================
"""


import torch
import torch.nn as nn



# ============================================================
# Import PIDGN-T Components
# ============================================================


from .encoders.genomic_autoencoder import (
    GenomicAutoencoder
)


from .encoders.resnet3d_encoder import (
    ResNet3DEncoder
)


from .encoders.lstm_clinical_encoder import (
    LSTMClinicalEncoder
)


from .graph.graph_construction import (
    GraphConstructionPipeline
)


from .fusion.cm_gaf import (
    CrossModalGraphAttentionFusion
)


from .temporal.tgnn import (
    TGNN
)


from .temporal.temporal_attention import (
    TemporalAttention
)





# ============================================================
# PIDGN-T Main Model
# ============================================================


class PIDGNT(nn.Module):


    """
    Complete PIDGN-T Architecture.


    Input Modalities:

        1. SNP genomic profiles

        2. Structural MRI volumes

        3. Longitudinal clinical observations


    Output:

        PD risk prediction probability

    """



    def __init__(

            self,

            snp_dim,

            clinical_dim,

            num_classes=2,


            genomic_embedding_dim=128,

            mri_embedding_dim=256,

            clinical_hidden_dim=128,


            fusion_dim=128,

            temporal_hidden_dim=256


    ):


        super().__init__()



        # ----------------------------------------------------
        # Modality-specific Encoders
        # ----------------------------------------------------


        self.genomic_encoder = (

            GenomicAutoencoder(

                input_dim=snp_dim,

                latent_dim=genomic_embedding_dim

            )

        )



        self.mri_encoder = (

            ResNet3DEncoder(

                embedding_dim=mri_embedding_dim

            )

        )



        self.clinical_encoder = (

            LSTMClinicalEncoder(

                input_dim=clinical_dim,

                hidden_dim=clinical_hidden_dim

            )

        )



        # ----------------------------------------------------
        # Graph Construction
        # ----------------------------------------------------


        self.graph_builder = (

            GraphConstructionPipeline()

        )



        # ----------------------------------------------------
        # Cross Modal Fusion
        # ----------------------------------------------------


        self.cm_gaf = (

            CrossModalGraphAttentionFusion(

                gene_dim=

                genomic_embedding_dim,


                brain_dim=

                mri_embedding_dim,


                hidden_dim=fusion_dim

            )

        )



        # ----------------------------------------------------
        # Temporal Graph Reasoning
        # ----------------------------------------------------


        self.tgnn = TGNN(

            input_dim=fusion_dim,

            hidden_dim=temporal_hidden_dim

        )


        self.temporal_attention = (

            TemporalAttention(

                hidden_dim=temporal_hidden_dim

            )

        )



        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------


        self.classifier = nn.Sequential(

            nn.Linear(

                temporal_hidden_dim,

                128

            ),


            nn.ReLU(),


            nn.Dropout(

                0.3

            ),


            nn.Linear(

                128,

                num_classes

            )

        )





    # ========================================================
    # Forward Propagation
    # ========================================================


    def forward(

            self,

            snp_data,

            mri_volume,

            clinical_sequence,

            graph_sequence,

            adjacency


    ):


        """
        Complete forward pass.


        Parameters
        ----------

        snp_data:

            SNP feature matrix


        mri_volume:

            3D MRI input


        clinical_sequence:

            Longitudinal clinical observations


        graph_sequence:

            Temporal fused graph sequence


        adjacency:

            Brain graph adjacency matrix



        Returns
        -------

        Prediction probabilities

        """



        # ----------------------------------------------------
        # 1. Feature Extraction
        # ----------------------------------------------------


        _, genomic_embedding = (

            self.genomic_encoder(

                snp_data

            )

        )



        brain_embedding = (

            self.mri_encoder(

                mri_volume

            )

        )



        clinical_embedding = (

            self.clinical_encoder(

                clinical_sequence

            )

        )



        # ----------------------------------------------------
        # 2. Graph Construction
        # ----------------------------------------------------


        graphs = (

            self.graph_builder.build_all_graphs(

                genomic_embedding,

                brain_embedding

            )

        )



        gene_graph = graphs[

            "gene_graph"

        ]["node_features"]



        brain_graph = graphs[

            "brain_graph"

        ]["node_features"]




        # ----------------------------------------------------
        # 3. CM-GAF Fusion
        # ----------------------------------------------------


        fusion_output = (

            self.cm_gaf(

                gene_graph,

                brain_graph

            )

        )


        fused_embedding = (

            fusion_output

            ["fusion_embedding"]

        )



        # ----------------------------------------------------
        # 4. Temporal Graph Learning
        # ----------------------------------------------------


        temporal_output = (

            self.tgnn(

                graph_sequence,

                adjacency

            )

        )



        temporal_states = (

            temporal_output

            ["sequence_features"]

        )



        attention_output = (

            self.temporal_attention(

                temporal_states

            )

        )



        temporal_embedding = (

            attention_output

            ["temporal_context"]

        )



        # ----------------------------------------------------
        # 5. Prediction
        # ----------------------------------------------------


        logits = (

            self.classifier(

                temporal_embedding

            )

        )


        probability = torch.softmax(

            logits,

            dim=1

        )



        return {


            "prediction":

                probability,


            "logits":

                logits,


            "fusion_attention":

                fusion_output

                ["attention_weights"],


            "temporal_attention":

                attention_output

                ["attention_weights"],


            "temporal_embedding":

                temporal_embedding


        }





if __name__ == "__main__":


    print(

        "PIDGN-T complete model architecture ready."

    )

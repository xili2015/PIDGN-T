"""
============================================================
PIDGN-T Ablation Study

Experiment:
    Remove TGNN Module

Purpose:
    Evaluate contribution of
    temporal graph reasoning.

Modification:
    TGNN replaced with static graph pooling.

============================================================
"""


import torch
import torch.nn as nn




class PIDGNT_NoTGNN(nn.Module):


    """
    PIDGN-T without temporal reasoning.

    Removes:

        TGNN
        Temporal Attention


    Uses:

        Static graph representation


    """



    def __init__(

            self,

            input_dim,

            hidden_dim=256,

            num_classes=2

    ):


        super().__init__()



        self.graph_encoder = nn.Sequential(

            nn.Linear(

                input_dim,

                hidden_dim

            ),

            nn.ReLU()

        )



        self.classifier = nn.Linear(

            hidden_dim,

            num_classes

        )




    def forward(

            self,

            graph_embedding

    ):


        """

        Static graph prediction.

        No longitudinal reasoning.

        """


        representation = self.graph_encoder(

            graph_embedding

        )


        logits = self.classifier(

            representation

        )



        return {


            "prediction":

                torch.softmax(

                    logits,

                    dim=1

                ),


            "temporal_removed":

                True

        }





if __name__ == "__main__":


    print(

        "Ablation model: TGNN removed."

    )

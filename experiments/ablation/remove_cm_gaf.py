"""
============================================================
PIDGN-T Ablation Study

Experiment:
    Remove CM-GAF Module

Purpose:
    Evaluate the contribution of
    Cross-Modal Graph Attention Fusion.

Modification:
    CM-GAF is replaced by simple feature concatenation.

============================================================
"""


import torch
import torch.nn as nn



class PIDGNT_NoCMGAF(nn.Module):


    """
    PIDGN-T without CM-GAF.


    Original:

    Gene Graph
          |
       CM-GAF
          |
    Brain Graph


    Ablated:

    Gene + Brain concatenation


    """



    def __init__(

            self,

            gene_dim,

            brain_dim,

            hidden_dim=256,

            num_classes=2

    ):


        super().__init__()



        self.fusion = nn.Sequential(

            nn.Linear(

                gene_dim + brain_dim,

                hidden_dim

            ),

            nn.ReLU(),

            nn.Dropout(0.3)

        )



        self.classifier = nn.Linear(

            hidden_dim,

            num_classes

        )




    def forward(

            self,

            gene_features,

            brain_features

    ):


        """

        Replace CM-GAF with
        direct feature concatenation.

        """


        fused = torch.cat(

            [

                gene_features,

                brain_features

            ],

            dim=-1

        )



        representation = self.fusion(

            fused

        )



        output = self.classifier(

            representation

        )


        return {


            "prediction":

                torch.softmax(

                    output,

                    dim=1

                ),


            "fusion_removed":

                True

        }





if __name__ == "__main__":


    print(

        "Ablation model: CM-GAF removed."

    )

"""
============================================================
PIDGN-T Ablation Study

Experiment:
    Remove Explainability Modules

Purpose:
    Evaluate effect of explainability
    mechanisms on model performance.

Removed:

    - SHAP-GNN
    - Grad-CAM
    - Temporal Saliency

============================================================
"""


import torch
import torch.nn as nn




class PIDGNT_NoExplainability(nn.Module):


    """
    PIDGN-T without explainability modules.

    The prediction pipeline remains unchanged,
    but explanation generation is disabled.

    """



    def __init__(

            self,

            embedding_dim,

            num_classes=2

    ):


        super().__init__()



        self.classifier = nn.Sequential(

            nn.Linear(

                embedding_dim,

                128

            ),

            nn.ReLU(),


            nn.Dropout(0.3),


            nn.Linear(

                128,

                num_classes

            )

        )




    def forward(

            self,

            temporal_embedding

    ):


        logits = self.classifier(

            temporal_embedding

        )



        return {


            "prediction":

                torch.softmax(

                    logits,

                    dim=1

                ),


            "explainability_disabled":

                True

        }





if __name__ == "__main__":


    print(

        "Ablation model: Explainability removed."

    )

"""
============================================================
Weighted Cross Entropy Loss

Purpose:
    Handle class imbalance between
    Parkinson's disease and control subjects.

============================================================
"""


import torch
import torch.nn as nn




class WeightedCrossEntropy(nn.Module):


    def __init__(
            self,
            class_weights
    ):

        super().__init__()


        self.weights = torch.tensor(

            class_weights,

            dtype=torch.float

        )



    def forward(

            self,

            predictions,

            labels

    ):


        criterion = nn.CrossEntropyLoss(

            weight=self.weights.to(

                predictions.device

            )

        )


        return criterion(

            predictions,

            labels

        )

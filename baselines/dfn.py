"""
============================================================
Baseline: Deep Fusion Network (DFN)

Purpose:
    Deep multimodal feature fusion using
    fully connected representation learning.

============================================================
"""


import torch
import torch.nn as nn



class DFN(nn.Module):


    def __init__(
            self,
            input_dim,
            classes=2
    ):

        super().__init__()



        self.network = nn.Sequential(

            nn.Linear(

                input_dim,

                256

            ),

            nn.ReLU(),


            nn.Dropout(
                0.3
            ),


            nn.Linear(

                256,

                128

            ),

            nn.ReLU(),


            nn.Linear(

                128,

                classes

            )

        )



    def forward(
            self,
            x
    ):

        return self.network(
            x
        )

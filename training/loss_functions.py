"""
============================================================
PIDGN-T Loss Functions

Implements:

L_total =
L_cls +
lambda1 L_align +
lambda2 L_temp +
lambda3 L_attn +
lambda4 L_reg

============================================================
"""


import torch
import torch.nn as nn



class PIDGNLoss(nn.Module):


    def __init__(

            self,

            lambda_align=0.1,

            lambda_temp=0.1,

            lambda_attn=0.05,

            lambda_reg=0.001

    ):


        super().__init__()



        self.lambda_align = lambda_align

        self.lambda_temp = lambda_temp

        self.lambda_attn = lambda_attn

        self.lambda_reg = lambda_reg



        self.classification_loss = (

            nn.CrossEntropyLoss()

        )



    def forward(

            self,

            prediction,

            labels,

            outputs

    ):



        L_cls = (

            self.classification_loss(

                prediction,

                labels

            )

        )



        L_align = outputs.get(

            "alignment_loss",

            0

        )



        L_temp = outputs.get(

            "temporal_loss",

            0

        )



        L_attn = outputs.get(

            "attention_loss",

            0

        )



        L_reg = outputs.get(

            "regularization_loss",

            0

        )



        total_loss = (

            L_cls

            +

            self.lambda_align * L_align

            +

            self.lambda_temp * L_temp

            +

            self.lambda_attn * L_attn

            +

            self.lambda_reg * L_reg

        )



        return total_loss

"""
============================================================
PIDGN-T
Temporal Attention Module

Purpose:
    Learn importance weights over longitudinal visits
    for Parkinson's disease progression modeling.

Input:
    TGNN hidden states over time

Output:
    Weighted temporal representation

Corresponds to:
    Section 3.5.2 Temporal Attention

============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class TemporalAttention(nn.Module):


    """
    Attention mechanism over longitudinal visits.

    The module assigns higher weights to clinically
    informative time points.
    """



    def __init__(
            self,
            hidden_dim,
            attention_dim=128
    ):

        super().__init__()



        self.W = nn.Linear(

            hidden_dim,

            attention_dim

        )


        self.context_vector = nn.Linear(

            attention_dim,

            1,

            bias=False

        )



    def forward(

            self,

            temporal_states

    ):

        """
        Parameters
        ----------
        temporal_states:

            Batch x Time x Hidden


        Returns
        -------
        Context vector
        Attention weights

        """



        energy = torch.tanh(

            self.W(

                temporal_states

            )

        )



        scores = self.context_vector(

            energy

        ).squeeze(-1)



        attention_weights = F.softmax(

            scores,

            dim=1

        )



        context_vector = torch.sum(

            temporal_states *
            attention_weights.unsqueeze(-1),

            dim=1

        )



        return {

            "temporal_context":

                context_vector,


            "attention_weights":

                attention_weights

        }



if __name__ == "__main__":

    print(
        "Temporal Attention module ready."
    )

"""
============================================================
PIDGN-T
Clinical Temporal Encoder

Purpose:
    Learn longitudinal representations from
    clinical and behavioral sequences.

Input:
    Temporal clinical observations

Output:
    Clinical temporal embeddings

Corresponds to:
    Section 3.2.3 Clinical Behavioral Encoder

============================================================
"""


import torch
import torch.nn as nn



class LSTMClinicalEncoder(
        nn.Module
):

    """
    Bidirectional LSTM encoder for longitudinal
    Parkinson's disease progression modeling.
    """



    def __init__(
            self,
            input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
    ):

        super().__init__()



        self.lstm = nn.LSTM(

            input_size=input_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout,

            bidirectional=True

        )



        self.projection = nn.Linear(

            hidden_dim * 2,

            hidden_dim

        )



    def forward(
            self,
            x
    ):

        """
        Parameters
        ----------
        x:

        Batch x Time x Clinical_Features


        Returns
        -------
        Temporal clinical embedding
        """

        outputs, (
            hidden,
            cell
        ) = self.lstm(
            x
        )


        # Combine forward and backward states

        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]


        combined = torch.cat(

            [
                forward_hidden,
                backward_hidden
            ],

            dim=1

        )


        embedding = self.projection(
            combined
        )


        return embedding



if __name__ == "__main__":

    print(
        "Clinical LSTM encoder ready."
    )

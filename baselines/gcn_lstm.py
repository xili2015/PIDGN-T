"""
============================================================
Baseline: GCN-LSTM

Purpose:
    Combine graph convolution with recurrent
    temporal modeling.

============================================================
"""


import torch
import torch.nn as nn



class GCNLayer(nn.Module):


    def __init__(
            self,
            input_dim,
            output_dim
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim
        )


    def forward(
            self,
            x,
            adjacency
    ):

        return torch.relu(

            self.linear(

                torch.matmul(
                    adjacency,
                    x
                )

            )

        )



class GCNLSTM(nn.Module):


    def __init__(
            self,
            input_dim,
            hidden_dim,
            classes=2
    ):

        super().__init__()


        self.gcn = GCNLayer(
            input_dim,
            hidden_dim
        )


        self.lstm = nn.LSTM(

            hidden_dim,

            hidden_dim,

            batch_first=True

        )


        self.fc = nn.Linear(

            hidden_dim,

            classes

        )



    def forward(
            self,
            graph_sequence,
            adjacency
    ):


        temporal_features=[]


        for t in range(
            graph_sequence.size(1)
        ):

            h = self.gcn(

                graph_sequence[:,t],

                adjacency

            )


            temporal_features.append(
                h.mean(dim=1)
            )


        x=torch.stack(
            temporal_features,
            dim=1
        )


        _,(h,_)=self.lstm(x)


        return self.fc(
            h[-1]
        )

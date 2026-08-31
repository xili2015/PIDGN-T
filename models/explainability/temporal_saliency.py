"""
============================================================
PIDGN-T
Temporal Saliency Explainability Module

Purpose:
    Analyze temporal contribution of longitudinal
    patient visits.

Input:
    TGNN hidden states

Output:
    Temporal importance scores

Corresponds to:
    Section 3.5.3 Temporal Interpretability
============================================================
"""


import torch



class TemporalSaliency:



    def __init__(
            self,
            model
    ):

        self.model = model



    def compute_saliency(
            self,
            temporal_states
    ):

        """
        Calculate gradient-based temporal importance.

        Score:

        S_t = |dP(y)/dH_t|

        """

        temporal_states.requires_grad = True



        prediction = self.model(
            temporal_states
        )


        probability = prediction[:,1]


        gradients = torch.autograd.grad(

            probability.sum(),

            temporal_states,

            retain_graph=True

        )[0]



        saliency = torch.abs(
            gradients
        )



        temporal_scores = torch.mean(

            saliency,

            dim=-1

        )


        return temporal_scores



    def rank_visits(
            self,
            temporal_scores
    ):

        """
        Rank clinically important visits.
        """

        ranking = torch.argsort(

            temporal_scores,

            descending=True

        )


        return ranking



if __name__ == "__main__":

    print(
        "Temporal saliency module ready."
    )

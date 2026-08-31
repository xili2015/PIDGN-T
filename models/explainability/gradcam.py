"""
============================================================
PIDGN-T
Grad-CAM Explainability Module

Purpose:
    Generate spatial explanations for MRI-based
    predictions.

Input:
    MRI feature maps from 3D CNN encoder

Output:
    3D activation heatmaps

Corresponds to:
    MRI Explainability using Grad-CAM
============================================================
"""


import torch
import torch.nn.functional as F



class GradCAM3D:


    def __init__(
            self,
            model,
            target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None


        self.register_hooks()



    def register_hooks(
            self
    ):

        """
        Register forward and backward hooks.
        """

        def forward_hook(
                module,
                input,
                output
        ):

            self.activations = output



        def backward_hook(
                module,
                grad_input,
                grad_output
        ):

            self.gradients = (
                grad_output[0]
            )


        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_backward_hook(
            backward_hook
        )



    def generate(
            self,
            input_volume,
            target_class
    ):

        """
        Generate Grad-CAM heatmap.
        """

        prediction = self.model(
            input_volume
        )


        score = prediction[
            target_class
        ]


        self.model.zero_grad()


        score.backward()



        gradients = (
            self.gradients
        )


        activations = (
            self.activations
        )


        weights = torch.mean(
            gradients,
            dim=(2,3,4),
            keepdim=True
        )


        cam = torch.sum(

            weights *
            activations,

            dim=1

        )


        cam = F.relu(
            cam
        )


        cam = (
            cam -
            cam.min()
        ) / (
            cam.max()
            -
            cam.min()
            +
            1e-8
        )


        return cam



if __name__ == "__main__":

    print(
        "3D Grad-CAM module ready."
    )

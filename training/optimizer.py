"""
============================================================
PIDGN-T Optimizer Configuration

Supports:
    AdamW optimization
    Learning rate scheduling

============================================================
"""


import torch



def build_optimizer(

        model,

        learning_rate=1e-4,

        weight_decay=1e-5

):


    """
    Create optimizer for PIDGN-T.
    """


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay

    )


    return optimizer





def build_scheduler(

        optimizer,

        patience=10

):


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        patience=patience

    )


    return scheduler

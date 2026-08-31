"""
============================================================
PIDGN-T Random Seed Controller

Purpose:
    Ensure reproducible experiments across
    multiple independent runs.

============================================================
"""


import random
import numpy as np
import torch



def set_seed(
        seed=42
):

    """
    Fix random seeds for reproducibility.
    """


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    torch.cuda.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.deterministic = True


    torch.backends.cudnn.benchmark = False



    print(

        f"Random seed fixed to {seed}"

    )





if __name__ == "__main__":


    set_seed(42)

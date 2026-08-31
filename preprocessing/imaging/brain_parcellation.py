"""
============================================================
PIDGN-T
Brain Parcellation Module

Purpose:
    Generate anatomical brain regions for
    connectome graph construction.

Supported Atlases:
    - AAL
    - Desikan-Killiany

Output:
    Brain region masks

============================================================
"""


import numpy as np



class BrainParcellation:


    def __init__(
            self,
            atlas="AAL"
    ):

        self.atlas = atlas



    def load_atlas(self):

        """
        Load anatomical brain atlas.

        In real experiments,
        atlas labels are obtained from
        standard neuroimaging libraries.
        """

        print(
            f"Loading {self.atlas} atlas"
        )


        return None



    def extract_regions(
            self,
            mri_volume,
            atlas_labels
    ):

        """
        Extract regional MRI representations.

        Each brain region becomes one
        graph node in the connectome graph.
        """

        regions = {}


        unique_labels = np.unique(
            atlas_labels
        )


        for label in unique_labels:

            if label == 0:
                continue


            region_mask = (
                atlas_labels == label
            )


            regions[label] = (
                mri_volume *
                region_mask
            )


        return regions



def create_brain_nodes(
        mri_volume,
        atlas_labels
):

    """
    Convert MRI regions into graph nodes.
    """

    parser = BrainParcellation()

    regions = parser.extract_regions(
        mri_volume,
        atlas_labels
    )


    nodes = []


    for region_id, region_data in regions.items():

        node = {

            "region_id":
                region_id,

            "features":
                region_data.mean()

        }


        nodes.append(node)



    return nodes



if __name__ == "__main__":

    print(
        "Brain parcellation module ready."
    )

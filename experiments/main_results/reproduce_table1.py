"""
============================================================
PIDGN-T

Reproduce Table 1:
Comparative Summary of PD Prediction Methods

Purpose:
    Generate the literature comparison table
    reported in the manuscript.

============================================================
"""


import pandas as pd





def create_table1():


    data = {


        "Method":[

            "Multi-Omics Ensemble",
            "3D CNN",
            "CNN-LSTM",
            "AdaMedGraph",
            "CNN-Transformer",
            "PIDGN",
            "HAMF",
            "RACF",
            "Deep Fusion CNN",
            "Transparent CNN",
            "DySAT",
            "EvolveGCN",
            "PIDGN-T"

        ],



        "Modalities":[

            "Multi-omics",
            "MRI",
            "Multimodal Time-Series",
            "MRI+Clinical+Genetic",
            "MRI+Clinical",
            "SNP+sMRI",
            "MRI+Speech+Gait",
            "SNP+MRI",
            "SNP+MRI+Clinical",
            "MRI+Wearable+Clinical",
            "Dynamic Graph",
            "Dynamic Graph",
            "SNP+sMRI+Clinical"

        ],



        "Architecture":[

            "ML Ensemble",
            "3D CNN",
            "CNN-LSTM",
            "Graph TGNN",
            "CNN Transformer",
            "Attention Fusion CNN",
            "Hierarchical Attention",
            "Residual Attention",
            "Deep Fusion CNN",
            "Transparent CNN",
            "Structural Temporal Attention",
            "Evolving GCN",
            "CM-GAF + TGNN"

        ],



        "Explainability":[

            "Feature Importance",
            "Grad-CAM",
            "SHAP/LIME",
            "SHAP",
            "Grad-CAM",
            "SHAP+Grad-CAM",
            "SHAP-CAM",
            "Grad-CAM",
            "SHAP+LIME",
            "Heatmaps",
            "No",
            "No",
            "SHAP-GNN+Grad-CAM"

        ]

    }


    table = pd.DataFrame(
        data
    )


    table.to_csv(

        "results/tables/table1_reproduced.csv",

        index=False

    )


    return table





if __name__ == "__main__":


    table=create_table1()


    print(table)

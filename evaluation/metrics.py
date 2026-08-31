"""
============================================================
PIDGN-T Evaluation Metrics

Purpose:
    Calculate evaluation metrics for PD prediction.

Metrics:
    Accuracy
    AUC
    Precision
    Recall
    F1-score
    Sensitivity
    Specificity

============================================================
"""


import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



def calculate_accuracy(
        y_true,
        y_pred
):

    return accuracy_score(
        y_true,
        y_pred
    )



def calculate_auc(
        y_true,
        probabilities
):

    return roc_auc_score(
        y_true,
        probabilities
    )



def calculate_precision(
        y_true,
        y_pred
):

    return precision_score(
        y_true,
        y_pred
    )



def calculate_recall(
        y_true,
        y_pred
):

    return recall_score(
        y_true,
        y_pred
    )



def calculate_f1(
        y_true,
        y_pred
):

    return f1_score(
        y_true,
        y_pred
    )



def calculate_sensitivity_specificity(
        y_true,
        y_pred
):

    """
    Calculate:

    Sensitivity = TP/(TP+FN)

    Specificity = TN/(TN+FP)

    """


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()


    sensitivity = tp / (
        tp + fn + 1e-8
    )


    specificity = tn / (
        tn + fp + 1e-8
    )


    return sensitivity, specificity



def evaluate_all(
        y_true,
        y_pred,
        probabilities
):


    sensitivity, specificity = (
        calculate_sensitivity_specificity(
            y_true,
            y_pred
        )
    )


    return {

        "Accuracy":
            calculate_accuracy(
                y_true,
                y_pred
            ),

        "AUC":
            calculate_auc(
                y_true,
                probabilities
            ),

        "Precision":
            calculate_precision(
                y_true,
                y_pred
            ),

        "Recall":
            calculate_recall(
                y_true,
                y_pred
            ),

        "F1-score":
            calculate_f1(
                y_true,
                y_pred
            ),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity

    }

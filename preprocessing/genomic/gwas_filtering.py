"""
============================================================
PIDGN-T
GWAS-based SNP Feature Filtering Module

Purpose:
    Select biologically relevant SNP features before
    genomic representation learning.

Pipeline:
    Raw SNP matrix
          |
          v
    GWAS significance filtering
          |
          v
    Selected SNP representation

Input:
    SNP genotype matrix
    GWAS association scores

Output:
    Reduced SNP feature matrix

============================================================
"""


import pandas as pd
import numpy as np



def load_snp_data(path):
    """
    Load raw SNP genotype matrix.

    Parameters
    ----------
    path : str
        Path to SNP data file

    Returns
    -------
    pandas.DataFrame
    """

    return pd.read_csv(path)



def load_gwas_scores(path):
    """
    Load GWAS association statistics.

    Required columns:

    SNP_ID
    P_VALUE
    EFFECT_SIZE

    """

    return pd.read_csv(path)



def filter_by_gwas(
        snp_data,
        gwas_data,
        p_threshold=0.05
):

    """
    Select SNPs according to GWAS significance.

    Parameters
    ----------
    snp_data :
        Original SNP matrix

    gwas_data :
        GWAS association information

    p_threshold :
        Statistical significance threshold


    Returns
    -------
    Filtered SNP matrix
    """

    significant_snps = gwas_data[
        gwas_data["P_VALUE"] <= p_threshold
    ]["SNP_ID"]


    selected_features = [
        snp for snp in significant_snps
        if snp in snp_data.columns
    ]


    return snp_data[selected_features]



def remove_low_variance_snps(
        snp_data,
        threshold=0.01
):

    """
    Remove SNPs with low variability.

    Low variance features provide limited
    discriminative information.
    """

    variance = snp_data.var()

    selected = variance[
        variance > threshold
    ].index


    return snp_data[selected]



def gwas_preprocessing(
        snp_path,
        gwas_path
):

    """
    Complete GWAS filtering pipeline.
    """

    snp_data = load_snp_data(
        snp_path
    )


    gwas_data = load_gwas_scores(
        gwas_path
    )


    snp_data = filter_by_gwas(
        snp_data,
        gwas_data
    )


    snp_data = remove_low_variance_snps(
        snp_data
    )


    return snp_data



if __name__ == "__main__":

    print(
        "GWAS SNP filtering module ready."
    )

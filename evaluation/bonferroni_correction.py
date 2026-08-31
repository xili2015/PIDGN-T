"""
============================================================
PIDGN-T Bonferroni Correction

Purpose:
    Control family-wise error rate in
    multiple statistical comparisons.

============================================================
"""





def bonferroni_adjustment(

        alpha,

        number_of_tests

):


    """

    Adjust significance threshold.

    alpha_corrected = alpha / m

    """


    return alpha / number_of_tests





def apply_correction(

        p_values,

        alpha=0.05

):


    """

    Apply Bonferroni correction.

    """

    m = len(
        p_values
    )


    corrected_alpha = (

        bonferroni_adjustment(

            alpha,

            m

        )

    )


    results=[]


    for p in p_values:


        results.append({

            "p_value":

                p,


            "significant":

                p < corrected_alpha

        })



    return results, corrected_alpha

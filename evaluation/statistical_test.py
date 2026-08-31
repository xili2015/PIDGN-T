"""
============================================================
PIDGN-T Statistical Testing

Purpose:
    Compare PIDGN-T against baseline models.

Method:
    Paired t-test across cross-validation folds.

============================================================
"""


from scipy.stats import ttest_rel





def paired_t_test(

        pidgn_scores,

        baseline_scores

):


    """
    Perform paired t-test.

    Input:

    Five-fold performance values

    """



    statistic, p_value = ttest_rel(

        pidgn_scores,

        baseline_scores

    )


    return {

        "t_statistic":
            statistic,

        "p_value":
            p_value

    }




def compare_models(

        pidgn_results,

        baseline_results

):


    results={}


    for name, scores in baseline_results.items():


        results[name]=paired_t_test(

            pidgn_results,

            scores

        )


    return results

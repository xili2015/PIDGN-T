"""
============================================================
PIDGN-T Inference Efficiency Analysis

Purpose:
    Measure patient-level inference latency.

Reported metric:
    Seconds per patient

============================================================
"""


import time

import torch





def measure_inference_time(

        model,

        sample,

        device,

        repetitions=10

):


    model.eval()


    times=[]



    with torch.no_grad():


        for _ in range(
            repetitions
        ):


            start=time.time()



            model(

                sample["snp"].to(device),

                sample["mri"].to(device),

                sample["clinical"].to(device),

                sample["graph"].to(device),

                sample["adjacency"].to(device)

            )


            end=time.time()



            times.append(

                end-start

            )



    return sum(times)/len(times)





def compare_latency(

        models,

        sample,

        device

):


    results={}


    for name, model in models.items():


        results[name]=measure_inference_time(

            model,

            sample,

            device

        )


    return results

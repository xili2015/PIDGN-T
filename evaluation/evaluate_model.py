"""
============================================================
PIDGN-T Model Evaluation Pipeline

Purpose:
    Evaluate trained PIDGN-T model
    and generate performance statistics.

============================================================
"""


import torch


from evaluation.metrics import evaluate_all





def evaluate_model(
        model,
        test_loader,
        device
):


    model.eval()


    y_true=[]

    y_pred=[]

    probabilities=[]



    with torch.no_grad():


        for batch in test_loader:


            output = model(

                batch["snp"].to(device),

                batch["mri"].to(device),

                batch["clinical"].to(device),

                batch["graph"].to(device),

                batch["adjacency"].to(device)

            )



            probs = output["prediction"][:,1]


            prediction = torch.argmax(
                output["prediction"],
                dim=1
            )



            y_true.extend(
                batch["label"].cpu().numpy()
            )


            y_pred.extend(
                prediction.cpu().numpy()
            )


            probabilities.extend(
                probs.cpu().numpy()
            )



    results = evaluate_all(

        y_true,

        y_pred,

        probabilities

    )


    return results



if __name__ == "__main__":

    print(
        "Evaluation pipeline ready."
    )

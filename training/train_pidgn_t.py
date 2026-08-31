"""
============================================================
PIDGN-T
Training Pipeline

Purpose:
    Train the complete PIDGN-T framework.

Includes:
    - Forward propagation
    - Multi-component loss optimization
    - Gradient update
    - Validation monitoring

Corresponds to:
    Section 3.7 Optimization Objective

============================================================
"""


import torch
from torch.utils.data import DataLoader



from models.pidgn_t import PIDGNT

from training.loss_functions import PIDGNLoss

from training.optimizer import build_optimizer

from training.seed_control import set_seed





def train_one_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        device
):

    """
    Train model for one epoch.
    """

    model.train()

    total_loss = 0



    for batch in dataloader:


        snp = batch["snp"].to(device)

        mri = batch["mri"].to(device)

        clinical = batch["clinical"].to(device)

        graph_sequence = batch["graph"].to(device)

        adjacency = batch["adjacency"].to(device)

        labels = batch["label"].to(device)



        optimizer.zero_grad()



        outputs = model(

            snp,

            mri,

            clinical,

            graph_sequence,

            adjacency

        )



        prediction = outputs["prediction"]



        loss = criterion(

            prediction,

            labels,

            outputs

        )



        loss.backward()



        optimizer.step()



        total_loss += loss.item()



    return total_loss / len(dataloader)





def validate(
        model,
        dataloader,
        device
):

    """
    Validation procedure.
    """

    model.eval()


    correct = 0

    total = 0



    with torch.no_grad():


        for batch in dataloader:


            outputs = model(

                batch["snp"].to(device),

                batch["mri"].to(device),

                batch["clinical"].to(device),

                batch["graph"].to(device),

                batch["adjacency"].to(device)

            )


            prediction = torch.argmax(

                outputs["prediction"],

                dim=1

            )


            labels = batch["label"].to(device)



            correct += (

                prediction == labels

            ).sum().item()


            total += labels.size(0)



    return correct / total





def train_pidgn_t(
        model,
        train_loader,
        val_loader,
        epochs,
        optimizer,
        criterion,
        device
):


    """
    Main training loop.
    """



    history = {


        "train_loss":[],

        "val_accuracy":[]


    }



    for epoch in range(epochs):


        loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device

        )


        accuracy = validate(

            model,

            val_loader,

            device

        )



        history["train_loss"].append(loss)

        history["val_accuracy"].append(accuracy)



        print(

            f"Epoch {epoch+1}/{epochs} "

            f"Loss={loss:.4f} "

            f"Val Acc={accuracy:.4f}"

        )



    return history





if __name__ == "__main__":


    set_seed(42)


    print(

        "PIDGN-T training pipeline ready."

    )

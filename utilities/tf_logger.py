"""
Module including Tensorboard logging utilities..

Created on 27/12/2022 at 10:18:00
@author: Filippo Torresan
"""

# Pytorch libraries imports
import torch
from torch.utils.tensorboard import SummaryWriter


def logs(metrics, writer, process, mode):
    """
     Function to log train or test data using Tensorboard.

    Inputs:

    - metrics (dict): dictionary storing saved metrics
    - writer (object): the Tensorboard summary writer
    - process (string): 'train' or 'test'
    - mode (string): 'supervised' or 'unsupervised'

    """

    # Storing free energy loss (scalar)
    writer.add_scalar("Loss/" + process, metrics["batch_avg_fe"], metrics["batch_num"])
    # Depending on the mode we log different things...
    if mode == "supervised":
        # Storing activity at the top layer together with datapoints and their labels
        writer.add_embedding(
            metrics["labels_units"],
            metadata=list(metrics["batch_labels"]),
            label_img=metrics["obs"],
            global_step=metrics["batch_num"],
            tag="labels_units_activity",
        )
        # Computing and storing train accuracy (for supervised mode only)
        writer.add_scalar(
            "Accuracy/" + process, metrics["correct_preds"], metrics["batch_num"]
        )

    else:

        # Hack to avoid Tensorflow complain when metrics["obs"] is not a proper image
        # NOTE: not sure if this now works with images...
        if len(metrics["obs"].size()) < 3:
            metrics["obs"] = None

        # Storing  bottom/input layer activity together with datapoints and their labels
        writer.add_embedding(
            metrics["input_activity"],
            metadata=list(metrics["batch_labels"]),
            label_img=metrics["obs"],
            global_step=metrics["batch_num"],
            tag="input_activity",
        )
        # Storing predictions to bottom/input layer together with datapoints and their labels
        writer.add_embedding(
            metrics["input_preds"],
            metadata=list(metrics["batch_labels"]),
            label_img=metrics["obs"],
            global_step=metrics["batch_num"],
            tag="input_preds",
        )
        # Storing activity at the top layer together with datapoints and their labels
        writer.add_embedding(
            metrics["top_units"],
            metadata=list(metrics["batch_labels"]),
            label_img=metrics["obs"],
            global_step=metrics["batch_num"],
            tag="top_activity",
        )
        # Storing activations at the top layer together with datapoints and their labels
        writer.add_embedding(
            metrics["top_activations"],
            metadata=list(metrics["batch_labels"]),
            label_img=metrics["obs"],
            global_step=metrics["batch_num"],
            tag="top_activations",
        )

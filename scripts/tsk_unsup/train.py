"""
Module defining the training function.

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Pytorch libraries imports
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from utilities import tf_logger
from utilities import ps_logger

# Global variables
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_pcn(data, pcn, num_batches, batch_size, exp_dir):
    """
    Function to train the PCN on a (batched) dataset.

    Inputs:

    - data (arrays): train dataset;
    - pcn (object): object of class pcn implementing the network;
    - num_itr (integer): iteration number;
    - num_batches (integer): number of batches the dataset was divided into;
    - batch_size (tuple): numbers of datapoint in each batch for training and testing (should be identical);
    - exp_dir (string): path name where results are saved.

    Outputs:

    - None.
    """

    # Setting default torch datatype
    torch.set_default_dtype(torch.float64)
    # PCN mode
    pcn_mode = pcn.mode
    # Retrieving size of training batches
    batch_size = batch_size[0]

    # In supervised mode convert bottom or top layer units into parameters
    if pcn.mode == "supervised":
        # In supervised mode we need to let the labels units free to update
        if pcn.labels_pos == "top":
            # Retrieving/setting number of classes
            num_classes = pcn.pcn_layers[-2].units.size()[0]
        else:
            # Retrieving/setting number of classes
            num_classes = pcn.pcn_layers[0].units.size()[0]
    # Defining the outer optimizer
    outer_opt = optim.Adam(pcn.select_params(inner=False), lr=pcn.lr)
    # Defining Pytorch dataloader
    tr_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # Creating custom logger
    tr_custom_wr = ps_logger.CustomLogger(
        num_batches, batch_size, pcn_mode, exp_dir + "/train_c"
    )
    # Tensorboard metrics logger/writer
    train_writer = SummaryWriter(exp_dir + "/train")
    # Sending network to GPU
    pcn.to(DEV)

    # Iterating over the batches
    for e in tqdm(range(num_batches)):

        print(" ")
        print(f"------------------------ Epoch {e} -------------------------")

        #### Data Retrieval and Processing ####

        # Retrieving next batch of observations and labels from iterable
        tr_obs, tr_labels = next(iter(tr_dataloader))
        # Sending batches to GPU
        tr_obs = tr_obs.to(DEV)
        tr_labels = tr_labels.to(DEV)
        # Defining a counter
        count_dp = 0

        # Printing original input size
        # print(f"Original input size: {tr_obs.size()}")
        # Transforming the input so that it is ready for the network
        # NOTE: we are transposing because....
        input = tr_obs.clone().squeeze().flatten(1, -1).T
        # Printing transformed input size
        # print(f"Transformed input size: {input.size()}")
        # Storing datapoint label
        curr_labl = tr_labels

        #### PCN Initialization and Inference ####

        # Activating the PCN network by sampling random unit activations
        pcn.apply(pcn.init_preds)

        if pcn.mode == "unsupervised":
            ### Passing in the input to the bottom units ###
            pcn.pcn_layers[0].set_units(input)
        else:
            ### In supervised mode, pass an input to the bottom units *and* top units ###
            # Preparing the one-hot-label
            one_hot_label = (torch.ones((num_classes, batch_size)) * 0.03).to(DEV)
            # one_hot_label[curr_labl] = 0.97
            one_hot_label[list(curr_labl), [list(torch.arange(batch_size))]] = 0.97
            # If the labels are given to the top, the images go to the bottom and vice versa
            if pcn.labels_pos == "top":
                # Passing input to bottom layer (say, image)
                pcn.pcn_layers[0].set_units(input)
                # Passing input to the top layer (say, label)
                pcn.pcn_layers[-2].set_units(one_hot_label)
            else:
                # Passing input to bottom layer (say, label)
                pcn.pcn_layers[0].set_units(one_hot_label)
                # Passing input to the top layer (say, image)
                pcn.pcn_layers[-2].set_units(input)

        # Start inference with the given input, outputting free energy, prediction to bottom layer
        # (input layer not relevant here), top unit activities/activations from *last* inferential cycle
        last_f, last_p, last_input_activity, last_tpu, last_tau, counter = pcn()

        #### PCN Learning ####

        # When datapoint in the batch has been processed, update parameters..
        pcn.learn(last_f, e, outer_opt)

        #### Metrics Computation/Logging ####

        # Computing and storing PCN-mode-dependent metrics
        correct_preds = None
        if pcn_mode == "supervised":

            if pcn.labels_pos == "top":
                labels_units = last_tpu.detach().clone().cpu().numpy().T
            else:
                labels_units = last_input_activity.detach().clone().cpu().numpy().T
            # print(f"Dimension of top units: {top_units.shape}")
            correct_preds = (
                (np.argmax(labels_units, axis=1) == list(curr_labl.cpu())).sum()
                / batch_size
                * 100.0
            )

            metrics = {
                "obs": tr_obs,
                "batch_num": e,
                "batch_labels": curr_labl.cpu().numpy(),
                "batch_avg_fe": last_f.detach().clone().cpu().numpy()[0] / batch_size,
                "labels_units": labels_units,
                "correct_preds": correct_preds,
            }

        else:

            metrics = {
                "obs": tr_obs,
                "batch_num": e,
                "batch_labels": curr_labl.cpu().numpy(),
                "batch_avg_fe": last_f.detach().clone().cpu().numpy()[0] / batch_size,
                "input_activity": last_input_activity.detach().clone().cpu().numpy().T,
                "input_preds": last_p.detach().clone().cpu().numpy().T,
                "top_units": last_tpu.detach().clone().cpu().numpy().T,
                "top_activations": last_tau.detach().clone().cpu().numpy().T,
            }

        # Tensorboard logging
        tf_logger.logs(metrics, train_writer, "train", pcn_mode)
        # Custom logging
        tr_custom_wr.log_all(metrics)

        # Printing some info during training
        print(" ")
        print("Training Info: ")
        print(" ")
        print(f"- Average free energy for batch {e}: {metrics['batch_avg_fe']}")
        print(f"- Accuracy for batch {e}: {correct_preds}")
        print(" ")

        #### Reset ####

        local = False
        pcn.reset(local)

    #### Save PCN ####

    print(" ")
    print("Training completed.")
    print("Saving network weights...")
    torch.save(pcn.state_dict(), exp_dir + "/net_ws")

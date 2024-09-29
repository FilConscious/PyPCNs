"""
Script to setup the PCN and loop over training/testing iterations (epochs).

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
from numpy.random import default_rng

# Pytorch libraries imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Custom packages/modules imports
from pcns import tm_pcn2
from scripts.tsk_sup import train
from scripts.tsk_sup import test
from utilities import dataset_wrp

# Global variable
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_task(params):
    """
    Function to train and test a PCN instance and save relevant metrics.

    Input:

    - params (dict): dictionary of relevant parameters.

    Outputs:

    - avg_ftra, avg_ftes (lists): free energy metrics from training and testing.
    """

    # Extracting parameters from dictionary
    ds_name = params["dataset_name"]
    dim_datapts = params["dim_datapts"]
    batch_size = params["batch_size"]
    num_batches = params["num_batches"]
    num_iters = params["num_iters"]
    layers = params["layers"]
    activ_func = params["activation_function"]
    eps = params["epsilon"]
    lr = params["learning_rate"]
    lr_sigmas = params["lr_sigmas"]
    lr_p = params["lr_priors"]
    dt = params["integration_step"]
    last_inf = params["inf_cycles"]
    target_dim = params["dim_target"]
    top_priors = params["top_priors"]
    top_sigma = params["top_sigma"]
    learn_sigmas = params["learn_sigmas"]
    fixed_input_activity = params["fixed_input_activity"]
    mode = params["mode"]
    labels_pos = params["labels_position"]
    process = params["process"]
    set_units = params["set_units"]
    num_transversals = params["num_transversals"]
    seed = params["seed"]
    data_dir = params["data_dir"]
    exp_dir = params["exp_dir"]
    saved_weights_dir = params["net_weights_dir"]
    exp_name = params["exp_name"]

    ### Defensive Programming ###
    if mode == "supervised":
        assert labels_pos == "top" or labels_pos == "bottom", print(
            "Invalid labels position in PCN supervised mode."
        )
        if labels_pos == "top":
            assert dim_datapts == layers[0], print(
                f"Input layer of size {layers[0]} cannot receive datapoint of size {dim_datapts}"
            )
        else:
            assert dim_datapts == layers[-1], print(
                f"Top layer of size {layers[0]} cannot receive datapoints of size {dim_datapts}"
            )

    # Creating dataset generation/loading object
    # NOTE: using a wrapper to pick and retrieve a Pytorch dataset dynamically or creating a new one
    data_retrvl = dataset_wrp.DataRetrieval(
        ds_name,
        data_dir,
        num_batches=150,
        batch_size=batch_size[0],
        set_units=set_units,
        num_transversals=num_transversals,
    )
    train_ds, test_ds = data_retrvl.retrieve()

    # Retrieving number of batches (from command line argument)
    num_batches_tr = num_batches[0]
    num_batches_ts = num_batches[1]

    print("")
    print("Warming up:")
    print(" ")

    print(
        f"Number of batches for training is {num_batches_tr}, for testing is {num_batches_ts}."
    )
    print(f"Using {DEV} for training and testing.")

    # Create PCN instance
    pc_net = tm_pcn2.TorchPcn(
        layers,
        activ_func,
        eps,
        batch_size[0],
        lr,
        lr_sigmas,
        lr_p,
        dt,
        last_inf,
        target_dim,
        top_priors,
        top_sigma,
        learn_sigmas,
        labels_pos,
        mode=mode,
        fixed_input_activity=fixed_input_activity,
    )

    for i in range(num_iters):

        print(" ")
        print(f"--------------------- Iteration {i} -----------------------")

        if process == "trts":
            # Training the network
            print("Training the network...")
            train.train_pcn(train_ds, pc_net, num_batches_tr, batch_size, exp_dir)
            # Testing the network
            print(" ")
            print("Testing the network...")
            test.test_pcn(test_ds, pc_net, num_batches_ts, batch_size, exp_dir)

        elif process == "tr":
            # Training the network
            print("Training the network...")
            train.train_pcn(train_ds, pc_net, num_batches_tr, batch_size, exp_dir)

        elif process == "ts":

            print("Testing the network...")
            test.test_pcn(
                test_ds, pc_net, num_batches_ts, batch_size, exp_dir, saved_weights_dir
            )

        else:

            raise ValueError(f"Not permitted value for variable process: {process}.")

    print("Done.")

    #### Defensive Programming ####
    # assert (
    #     len(last_ftra) == len(last_ftes) == num_iters
    # ), "You have not saved an average free energy per iteration."
    # assert (
    #     len(tr_preds) == len(ts_preds) == num_iters
    # ), "You have not saved a predictions array per iteration."
    # assert (
    #     len(ts_inl) == num_iters
    # ), "You have not saved a input activity list per iteration."
    #### End of DP ####

    # Loading network attributes
    # with open(exp_dir + "/net_attrs.json", "r") as rfile:
    #    pcn_attr = json.load(rfile)
    # Rewriting the
    # pcn.__dict__ = pcn_attr
    #
    # Saving network attributes
    # with open(exp_dir + "/net_attrs.json", "w") as wfile:
    #    json.dump(pcn.__dict__, wfile)

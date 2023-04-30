"""
Module defining the testing function.

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Pytorch libraries imports
import torch
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

# Custom imports
from utilities import tf_logger
from utilities import ps_logger

# Global variables
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_pcn(data, pcn, num_batches, batch_size, exp_dir, saved_ws_dir=None):
    """
    Function to test the PCN on a (batched) dataset.

    Inputs:

    - data (arrays): test dataset;
    - pcn (object): object of class pcn implementing the network;
    - num_itr (integer): iteration number;
    - num_batches (integer): number of batches the dataset was divided into;
    - batch_size (integer): number of datapoint in each batch;
    - exp_dir (string): path name where results are saved.

    Outputs:

    - None.
    """

    # Setting default torch datatype
    torch.set_default_dtype(torch.float64)
    # Shuffle parameter
    shuffle_ds = True
    # PCN mode
    pcn_mode = pcn.mode
    # Retrieve size of test batches
    batch_size = batch_size[1]
    # Load network weights if required
    if saved_ws_dir != None:
        print("Loading network weights...")
        print(f"Loading directory: {saved_ws_dir}/net_ws")
        pcn.load_state_dict(torch.load(saved_ws_dir + "/net_ws"))

    # In supervised mode convert bottom or top layer units into parameters
    if pcn.mode == "supervised":
        # In supervised mode we need to let the labels units free to update
        if pcn.labels_pos == "top":
            # Label units at the top
            pcn.pcn_layers[-2].fix_activity = (False, "supervised")
            # Retrieving/setting number of classes
            num_classes = pcn.pcn_layers[-2].units.size()[0]
        else:
            # Labels units at the bottom
            pcn.pcn_layers[0].fix_activity = (False, "supervised")
            # Retrieving/setting number of classes
            num_classes = pcn.pcn_layers[0].units.size()[0]

    elif pcn.mode == "unsupervised":
        # in unsupervised mode check if we want to set top level priors.
        if pcn.top_priors == True:
            # if yes, change fix_activity attribute to True
            pcn.pcn_layers[-2].fix_activity = (True, "unsupervised")
            # Shuffle dataset to False to plot transversals
            # NOTE: temporary fix to plot transversals, this should be handled by using the transversal
            # argument from CL and not the top prior attribute...
            shuffle_ds = False
        else:
            # do nothing
            pass

    # Retrieving target dimension
    target_dim = pcn.t_dim
    # Setting the number of inferential cycle for testing
    pcn.last_inf = pcn.inf_cycles_values[1]
    # Defining Pytorch dataloader
    ts_dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle_ds)
    ts_dataloader_iter = iter(ts_dataloader)
    # Creating custom logger
    ts_custom_wr = ps_logger.CustomLogger(
        num_batches, batch_size, pcn_mode, exp_dir + "/test_c"
    )
    # Tensorboard metrics logger/writer
    test_writer = SummaryWriter(exp_dir + "/test")

    # Sending network to GPU
    pcn.to(DEV)

    # Iterating over the batches
    for e in tqdm(range(num_batches)):

        print(" ")
        print(f"---------------------- Epoch {e} -----------------------")

        #### Data Retrieval and Processing ####

        # Retrieving next batch of observations and labels from iterable
        ts_obs, ts_labels = next(ts_dataloader_iter)
        print(f"Labels for epoch {e}: {ts_labels}")
        ts_obs = ts_obs.to(DEV)
        ts_labels = ts_labels.to(DEV)
        # Defining a counter
        count_dp = 0

        # Printing original input size
        # print(f"Original input size: {ts_obs.size()}")
        # Transforming the input so that it is ready for the network
        # NOTE: we are transposing because....

        input = ts_obs.clone().squeeze().flatten(1, -1).T
        # Printing transformed input size
        # print(f"Transformed input size: {input.size()}")
        # Storing datapoint label
        curr_labl = ts_labels

        #### PCN Initialization and Inference ####

        # Activating the PCN network by sampling random unit activations
        pcn.apply(pcn.init_preds)

        if pcn.mode == "unsupervised":
            # Unsupervised mode: passing complete or truncated input to bottom layer
            if pcn.top_priors:
                # With top priors set top layer to the input values...
                pcn.pcn_layers[-2].set_units(input)
                # ...and the bottom layer just to zero values
                pcn.pcn_layers[0].set_units(
                    torch.zeros_like(pcn.pcn_layers[0].act_units).to(DEV)
                )
            else:
                if target_dim != 0:
                    # Zeroing target_dim entries of the input (input corruption)
                    input[-target_dim:] = 0.0
                # Passing in the input to the bottom units
                pcn.pcn_layers[0].set_units(input)

        else:
            ### In supervised mode, pass an input to the bottom units *and* top units ###
            # Preparing the one-hot-label
            one_hot_label = (torch.ones((num_classes, batch_size)) * 0.03).to(DEV)
            # one_hot_label[curr_labl] = 0.97
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
        last_f, last_p, last_input_activity, last_tpu, last_tau, _ = pcn()

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
                "obs": ts_obs,
                "batch_num": e,
                "batch_labels": curr_labl.cpu().numpy(),
                "batch_avg_fe": last_f.detach().clone().cpu().numpy()[0] / batch_size,
                "labels_units": labels_units,
                "correct_preds": correct_preds,
            }

        else:

            metrics = {
                "obs": ts_obs,
                "batch_num": e,
                "batch_labels": curr_labl.cpu().numpy(),
                "batch_avg_fe": last_f.detach().clone().cpu().numpy()[0] / batch_size,
                "input_activity": last_input_activity.detach().clone().cpu().numpy().T,
                "input_preds": last_p.detach().clone().cpu().numpy().T,
                "top_units": last_tpu.detach().clone().cpu().numpy().T,
                "top_activations": last_tau.detach().clone().cpu().numpy().T,
            }

        # Tensorboard logging
        tf_logger.logs(metrics, test_writer, "test", pcn_mode)
        # Custom logging
        ts_custom_wr.log_all(metrics)

        # Printing some info during training
        print(" ")
        print("Test Info:")
        print(" ")
        print(f"- Average free energy for batch {e}: {metrics['batch_avg_fe']}")
        print(f"- Accuracy for batch {e}: {correct_preds}")
        print(" ")
        # print(f"------------------------ End --------------------------")

        #### Reset ####

        local = False
        pcn.reset(local)
        # Zeroing the gradients of the weights as they are being accumulated but not used during testing
        # NOTE: not doing this overloads the memory and leads to a frozen system
        pcn.zero_grad()

    print(" ")
    print("Testing completed.")

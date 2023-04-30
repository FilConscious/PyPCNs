"""
Module defining a custom logger class.

Created on 27/12/2022 at 11:05:00
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import numpy as np
import pandas as pd


class CustomLogger(object):
    """
    Custom logger class to store metrics from experiments with a PCN.
    """

    def __init__(self, num_batches, batch_size, pcn_mode, save_dir):
        """

        Inputs:

        - num_batches (integer): number of batches
        - batch_size (integer): number of datapoints in one batch
        - save_dir (string): directory where dataframes are stored
        """

        # Inputs saved as attributes
        self.num_batches = num_batches
        self.bs = batch_size
        self.pcn_mode = pcn_mode
        self.save_dir = save_dir
        # Creating storage attributes that span all the batches
        self.avg_fes = []
        self.correct_preds = []

    def log_avg_fe(self, batch_avg_fe):
        """
        Method logging batch average (last) free energies.

        For every datapoint in a batch the PCN reaches equilibrium.
        At that point, the last computed free energy is stored.

        Inputs:

        - last_fe (float): free energy;
        """

        # Creating specific saving directory
        logdir = self.save_dir + "/loss"
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        # Storing average free energy for current batch
        self.avg_fes.append(batch_avg_fe)

        # Save data into dataframe at the end of the batch
        if len(self.avg_fes) == self.num_batches:
            # Create Pandas dataframe
            df = pd.DataFrame(self.avg_fes, columns=["BAFE"])
            df.to_csv(logdir + "/bafe.csv")

    def log_accuracy(self, correct_preds):
        """
        Method logging the batch predictive accuracy (only to be used in supervised mode).

        Inputs:

        - correct_preds (float): percentage of correct predictions for current batch;
        """

        # Creating specific saving directory
        logdir = self.save_dir + "/accuracy"
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        # Storing average free energy for current batch
        self.correct_preds.append(correct_preds)

        # Save data into dataframe at the end of the batch
        if len(self.correct_preds) == self.num_batches:
            # Create Pandas dataframe
            df = pd.DataFrame(self.correct_preds, columns=["Accuracy"])
            df.to_csv(logdir + "/accuracy.csv")

    def log_layer(self, layer_activity, batch_labels, batch_num, layer_activity_ID):
        """
        Method logging some kind of PCN layer activity for every datapoint in the batch
        at the end of inference.

        The activity could correspond to:

        - predictions to bottom layer (in supervised or unsupervised mode)
        - top layer activity or activation (in supervised or unsupervised mode)
        - input layer activity (usually in unsupervised mode)

        Inputs:

        - input_preds (ndarray): PCN predictions directed to bottom layer
        - batch_labels (list): list of labels identifying the datapoints
        - batch_num (integer): batch number for current data
        - layer_activity_ID (string): the type of layer activity (see above)
        """

        # Defensive Programming
        assert layer_activity.shape[0] == batch_labels.shape[0], print(
            f"Number of input predictions, {layer_activity.shape[0]}, is not equal to number of labels, {batch_labels.shape[0]}"
        )

        # Creating specific saving directory
        logdir = self.save_dir + f"/{layer_activity_ID}"
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        # Concatenating datapoints labels and prediction array
        # NOTE: in this way the first column of the dataframe will identify the datapoint
        # for which we are recording the layer activity
        data = np.concatenate((batch_labels[:, None], layer_activity), axis=1)
        # Create Pandas dataframe with the predictions (and column of labels)
        df = pd.DataFrame(
            data,
            columns=["Label"] + [f"unit {i}" for i in range(layer_activity.shape[1])],
        )
        df.to_csv(logdir + f"/values_b{batch_num}.csv")

    def log_all(self, metrics):
        """
        Method performing logging of all relevant metrics based on PCN mode (supervised or unsupervised).

        Input:

        - metrics (dict): dictionary with relevant data saved from the batch

        """

        print(f"Logging metrics for PCN in {self.pcn_mode} mode...")
        # Logging free energy loss
        self.log_avg_fe(metrics["batch_avg_fe"])
        # Depending on the mode we log different things...
        if self.pcn_mode == "supervised":

            self.log_accuracy(metrics["correct_preds"])
            self.log_layer(
                metrics["labels_units"],
                metrics["batch_labels"],
                metrics["batch_num"],
                "labels_units",
            )

        else:

            self.log_layer(
                metrics["input_activity"],
                metrics["batch_labels"],
                metrics["batch_num"],
                "input_activity",
            )
            self.log_layer(
                metrics["input_preds"],
                metrics["batch_labels"],
                metrics["batch_num"],
                "input_preds",
            )
            self.log_layer(
                metrics["top_units"],
                metrics["batch_labels"],
                metrics["batch_num"],
                "top_units",
            )
            self.log_layer(
                metrics["top_activations"],
                metrics["batch_labels"],
                metrics["batch_num"],
                "top_activations",
            )

        print("...Done logging.")

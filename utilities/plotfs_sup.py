"""
Module defining plotting functions for PCN trained in *supervised* mode.

Created on 27/12/2022 at 18:48:00
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.font_manager

# NOTE: the Computer Modern font shipped with matplotlib is called 'cmss10' (sans-serif) or 'cmr10' (serif)
plt.rcParams["font.sans-serif"] = "cmss10"
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["axes.formatter.use_mathtext"] = True

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("axes", unicode_minus=False)


class SupPlotter(object):
    """
    Custom plotter class to plot stored metrics from experiments with a PCN.

    """

    def __init__(self, runs, runs_type, exp_id, vis_type, figures_dir):
        """
        Init method.

        Inputs:

        - runs (list): list of paths to store metrics (one for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - vis_type (string): type of visualization ("compare", "average", "comp_avg")
        - save_dir (string): directory where to save the plots"""

        # List of paths
        self.runs = runs
        # Dictionary of runs' parameters
        self.runs_params = {}
        # Retrieving runs parameters
        self.get_params(self.runs)

        ### Params from runs' config files ###
        # Mode (string) the PCN was trained/tested with (should be the same for every run)
        self.pcn_mode = self.runs_params["run_0"]["mode"]
        # batch_size (integer): number of datapoints per batch in saved experiments
        self.batch_size = self.runs_params["run_0"]["batch_size"]
        print(self.batch_size)

        ### Params CL ###
        # runs_type (string): whether we are looking at train/test or train and test data
        self.runs_type = runs_type
        # Experiment id
        self.exp_id = exp_id
        # Type of visualization to produce: compare, average, or comp_avg
        self.vis_type = vis_type
        # figures_dir (string): directory where to store generated figures
        self.figures_dir = figures_dir

    def get_params(self, runs):
        """
        Function that loads the configuration files and retrieve run parameters.
        """

        for n, r in enumerate(runs):
            # Configuration directory
            cfg_dir = r + "/config.json"
            # Opening the json file storing the parameters
            with open(cfg_dir, "r") as read_p:
                # Loading params
                cfg_params = json.load(read_p)

            self.runs_params[f"run_{n}"] = cfg_params

    def plot_metrics(self, runs):
        """
        Function to load metrics store in a series of experiments.

        Inputs:

        - runs (list): list of paths to the saved metrics

        """

        # List of runs types
        rtypes = []
        if self.runs_type == "trts":
            # runs_type indicate that we have train and test data
            rtypes.append(("train", self.batch_size[0]))
            rtypes.append(("test", self.batch_size[1]))
        else:
            # runs_type is either 'train' or 'test' so we have one type of data only
            rtypes.append(self.runs_type)

        # Retrieve stored metrics
        for t in rtypes:

            run_type, batch_size = t
            # Creating dataframes generator objects for training data
            accuracies_df = (
                pd.read_csv(r + f"/{run_type}_c/accuracy/accuracy.csv", index_col=0)
                for r in runs
            )
            losses_df = (
                pd.read_csv(r + f"/{run_type}_c/loss/bafe.csv", index_col=0)
                for r in runs
            )
            # top_activs = (pd.read_csv(r + f"/{t}_c/top_activations/top_activations.csv") for r in runs)
            # top_unitss = (pd.read_csv(r + f"/{t}_c/top_units/top_units.csv") for r in runs)
            # Calling plotting functions
            self.plt_bafe(
                list(losses_df),
                run_type,
                batch_size,
                self.vis_type,
                self.figures_dir,
            )
            self.plt_accuracy(
                list(accuracies_df),
                run_type,
                batch_size,
                self.vis_type,
                self.figures_dir,
            )

    def plt_bafe(self, avg_fes, run_type, batch_size, vis_type, save_dir):
        """
        Function to plot free energy metrics from training and testing.

        Inputs:

        - avg_fes (list): list of batch average free energy dataframes (one for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - vis_type (string): type of visualization ("compare", "average", "comp_avg")
        - save_dir (string): directory where to save the plots
        """

        # Creating figure
        fig, axs = plt.subplots(figsize=(6, 6))  # , sharex=True)
        # Process dataframes differently depending on vis_type
        if vis_type == "compare":
            # List to store legend labels
            legend_labels = []
            # Looping over runs dataframes
            for idx, data_fr in enumerate(avg_fes):
                # Retrieving value of self.exp_id for current run
                id_value = str(self.runs_params[f"run_{idx}"][self.exp_id])
                # Create shorter label for plots
                if self.exp_id == "learning_rate":
                    legend_labels.append((r"$\beta_{2} =$", id_value))
                elif self.exp_id == "lr_sigmas":
                    legend_labels.append((r"$\beta_{3} =$", id_value))
                elif self.exp_id == "top_sigma":
                    legend_labels.append((r"$v_{0}^{(1)} =$", id_value))
                else:
                    legend_labels.append((self.exp_id, id_value))
                # Plotting data for current run
                data_fr.plot.line(ax=axs)

            # Completing training axes
            axs.legend([f"{label}: ${v}$" for (label, v) in legend_labels])
            axs.grid(True)
            axs.set_xlabel("batch number")
            axs.set_ylabel("free energy")
            axs.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.yaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.set_title(f"Batch Average Free Energy ({run_type})")

        elif vis_type == "average":

            # Concatenating dataframes from different runs and converting to numpy
            all_runs = pd.concat(avg_fes, axis=1).to_numpy()
            # Computing mean, std etc for confidence intervals
            mean_bafes = np.mean(all_runs, axis=1)
            std_bafes = np.std(all_runs, axis=1)
            run_count = len(avg_fes)
            num_batches = all_runs.shape[0]
            low_ci = mean_bafes - 1.96 * std_bafes / np.sqrt(run_count)
            upp_ci = mean_bafes + 1.96 * std_bafes / np.sqrt(run_count)
            # Plotting
            axs.plot(np.arange(num_batches), mean_bafes, label="Mean BAFE")
            axs.fill_between(
                np.arange(num_batches), low_ci, upp_ci, facecolor="blue", alpha=0.3
            )
            # Completing training axes
            axs.legend()
            axs.grid(True)
            axs.set_xlabel("batch number")
            axs.set_ylabel("free energy")
            axs.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.yaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.set_title(f"Run Average of BAFE ({run_type})")

        else:

            raise NotImplementedError

        # Save figure
        plt.savefig(
            save_dir + "/" + f"{vis_type}_{run_type}_bafe.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # IMPORTANT: showing after saving otherwise closing the displayed image frees it from memory
        # and what you get saved is a blank slate
        plt.show()

    def plt_accuracy(self, accs, run_type, batch_size, vis_type, save_dir):
        """
        Function to plot free energy metrics from training and testing.

        Inputs:

        - accs (list): list of prediction accuracy dataframes (one for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - vis_type (string): type of visualization ("compare", "average", "comp_avg")
        - save_dir (string): directory where to save the plots
        """

        # Creating figure
        fig, axs = plt.subplots(figsize=(6, 6))  # , sharex=True)
        # Process dataframes differently depending on vis_type
        if vis_type == "compare":
            # List to store legend labels
            legend_labels = []
            # Looping over runs dataframes
            for idx, data_fr in enumerate(accs):
                # Retrieving value of self.exp_id for current run
                id_value = str(self.runs_params[f"run_{idx}"][self.exp_id])
                # Create shorter label for plots
                if self.exp_id == "learning_rate":
                    legend_labels.append((r"$\beta_{2} =$", id_value))
                elif self.exp_id == "lr_sigmas":
                    legend_labels.append((r"$\beta_{3} =$", id_value))
                elif self.exp_id == "top_sigma":
                    legend_labels.append((r"$v_{0}^{(1)} =$", id_value))
                else:
                    legend_labels.append((self.exp_id, id_value))

                # Plotting data for current run
                data_fr.plot.line(ax=axs)

            # Completing training axes
            axs.legend([f"{label}: ${v}$" for (label, v) in legend_labels])
            axs.grid(True)
            axs.set_xlabel("batch number")
            axs.set_ylabel("accuracy")
            axs.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.yaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.set_title(f"Batch Accuracy ({run_type})")

        elif vis_type == "average":

            # Concatenating dataframes from different runs and converting to numpy
            all_runs = pd.concat(accs, axis=1).to_numpy()
            # Computing mean, std etc for confidence intervals
            mean_accs = np.mean(all_runs, axis=1)
            std_accs = np.std(all_runs, axis=1)
            run_count = len(accs)
            num_batches = all_runs.shape[0]
            low_ci = (mean_accs - 1.96 * std_accs / np.sqrt(run_count)).squeeze()
            upp_ci = (mean_accs + 1.96 * std_accs / np.sqrt(run_count)).squeeze()
            # Plotting
            axs.plot(np.arange(num_batches), mean_accs, label="Mean Accuracy")
            axs.fill_between(
                np.arange(num_batches), low_ci, upp_ci, facecolor="blue", alpha=0.3
            )
            # Completing training axes
            axs.legend()
            axs.grid(True)
            axs.set_xlabel("batch number")
            axs.set_ylabel("accuracy")
            axs.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.yaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            axs.set_title(f"Run Average of Accuracy ({run_type})")

        else:

            raise NotImplementedError

        # Save figure
        plt.savefig(
            save_dir + "/" + f"{vis_type}_{run_type}_accs.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )

        # IMPORTANT: showing after saving otherwise closing the displayed image frees it from memory
        # and what you get saved is a blank slate
        plt.show()

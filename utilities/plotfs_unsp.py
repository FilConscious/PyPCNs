"""
Module defining plotting functions for PCN trained in *unsupervised* mode.

Created on 02/01/2023 at 21:13:00
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


class UnsPlotter(object):
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
        Function to load metrics stored in a series of experiments.

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
            batch_size = (
                self.batch_size[0] if self.runs_type == "train" else self.batch_size[1]
            )
            rtypes.append((self.runs_type, batch_size))

        # Retrieve stored metrics
        for i, t in enumerate(rtypes):

            run_type, batch_size = t
            num_batch_index = 0 if run_type == "train" else 1
            # Creating dataframes generator objects for training data
            input_preds_df = (
                pd.read_csv(
                    r + f"/{run_type}_c/input_preds/values_b{batch_num}.csv",
                    index_col=0,
                )
                for n, r in enumerate(runs)
                for batch_num in range(
                    self.runs_params[f"run_{n}"]["num_batches"][num_batch_index]
                )
            )
            top_units_df = (
                pd.read_csv(
                    r + f"/{run_type}_c/top_units/values_b{batch_num}.csv", index_col=0
                )
                for n, r in enumerate(runs)
                for batch_num in range(
                    self.runs_params[f"run_{n}"]["num_batches"][num_batch_index]
                )
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
            self.plt_input_preds(
                list(input_preds_df),
                run_type,
                batch_size,
                self.figures_dir,
            )
            # self.plt_input_preds_w_transv(
            #     list(input_preds_df),
            #     run_type,
            #     batch_size,
            #     self.figures_dir,
            # )
            self.plt_top_units(
                list(top_units_df),
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

    def plt_top_units(self, top_units, run_type, batch_size, vis_type, save_dir):
        """
        Function to plot top layer activity from training and testing.

        N.B. The argument vis_type (average, compare etc) has no effect below because we are just
        plotting the data for each run separately.

        Inputs:

        - top_units (list): list of top units activity dataframes (one for every batch for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - vis_type (string): type of visualization ("compare", "average", "comp_avg")
        - save_dir (string): directory where to save the plots
        """

        # Retrieving number of runs
        num_runs = len(self.runs)
        # Retrieving number of batches for each run
        runs_num_batches = [
            self.runs_params[f"run_{r}"]["num_batches"] for r in range(num_runs)
        ]
        # Looping over the runs
        for r in range(num_runs):
            # Retrieving value of num_batches parameter
            tr_num_batches = runs_num_batches[r][0]
            ts_num_batches = runs_num_batches[r][1]
            # Number of batches from previous runs
            tr_exclude_batches = sum(
                [runs_num_batches[prev_r][0] for prev_r in range(r)]
            )
            ts_exclude_batches = sum(
                [runs_num_batches[prev_r][1] for prev_r in range(r)]
            )
            # Plotting train top unit activity for run r
            if run_type == "train":
                # Selecting the batches only for current run
                run_top_units = top_units[tr_exclude_batches:]
                # Looping over parts of the total bacth number to see some progression (in training)
                parts = 4
                num_batches_part = tr_num_batches // parts
                for p in range(parts):
                    # Concatenating all batches dataframes of the same part of the total batches
                    batches_pn = pd.concat(
                        run_top_units[
                            p * num_batches_part : num_batches_part * (p + 1)
                        ],
                        axis=0,
                    ).to_numpy()

                    #### Creating figure
                    fig, axs = plt.subplots(figsize=(6, 6))  # , sharex=True)
                    # Creating axes for training data (activity)
                    scatter_top_units = axs.scatter(
                        batches_pn[:, 1],
                        batches_pn[:, 2],
                        c=batches_pn[:, 0],
                        cmap="tab10",
                    )

                    #### Completing axes with labels and legends ####

                    # Produce a legend with the unique colors from the scatter
                    legend = axs.legend(
                        *scatter_top_units.legend_elements(),
                        loc="upper right",
                        title="Classes",
                    )
                    axs.add_artist(legend)

                    axs.set_xlabel("x")
                    axs.set_ylabel("y")
                    # ax_tr.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
                    axs.set_title(f"Top units activity (training)")

                    # Save figure
                    plt.savefig(
                        save_dir + "/" + f"run{r}_clusters_{run_type}_p{p}.pdf",
                        format="pdf",
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )

                    # IMPORTANT: showing after saving otherwise closing the displayed image frees it from memory
                    # and what you get saved is a blank slate
                    plt.show()

            elif run_type == "test":
                # Selecting the batches only for current run
                run_top_units = top_units[ts_exclude_batches:]
                # Concatenating all batches dataframes saved during testing
                batches_pn = pd.concat(run_top_units, axis=0).to_numpy()

                #### Creating figure
                fig, axs = plt.subplots(figsize=(6, 6))  # , sharex=True)
                # Creating axes for training data (activity)
                scatter_top_units = axs.scatter(
                    batches_pn[:, 1],
                    batches_pn[:, 2],
                    c=batches_pn[:, 0],
                    cmap="tab10",
                )

                #### Completing axes with labels and legends ####

                # Produce a legend with the unique colors from the scatter
                legend = axs.legend(
                    *scatter_top_units.legend_elements(),
                    loc="upper right",
                    title="Classes",
                )
                axs.add_artist(legend)

                axs.set_xlabel("x")
                axs.set_ylabel("y")
                # ax_tr.xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
                axs.set_title(f"Top units activity (testing)")

                # Save figure
                plt.savefig(
                    save_dir + "/" + f"run{r}_clusters_{run_type}_all.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

                # IMPORTANT: showing after saving otherwise closing the displayed image frees it from memory
                # and what you get saved is a blank slate
                plt.show()

    def plt_input_preds(self, input_preds, run_type, batch_size, save_dir):
        """
        Function to plot predictions to bottom layer either from training or testing.

        Inputs:

        - input_preds (list): list of input predictions dataframes (one for every batch for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - save_dir (string): directory where to save the plots
        """

        # Hard-coded number of batches for which to plot input predictions (images)
        batches_to_plot = 4
        # Retrieving number of runs
        num_runs = len(self.runs)
        # Retrieving number of batches for each run
        runs_num_batches = [
            self.runs_params[f"run_{r}"]["num_batches"] for r in range(num_runs)
        ]
        # Looping over the runs
        for r in range(num_runs):
            # Retrieving value of num_batches parameter
            tr_num_batches = runs_num_batches[r][0]
            ts_num_batches = runs_num_batches[r][1]
            # Number of batches from previous runs
            tr_exclude_batches = sum(
                [runs_num_batches[prev_r][0] for prev_r in range(r)]
            )
            ts_exclude_batches = sum(
                [runs_num_batches[prev_r][1] for prev_r in range(r)]
            )
            # Plotting train top unit activity for run r
            if run_type == "train":
                # Selecting the batches only for current run
                run_input_preds = input_preds[tr_exclude_batches:]
                # Looping over parts of the total bacth number to see some progression (in training)
                parts = 4
                num_batches_part = tr_num_batches // parts
                fraction_batches_to_plot = batches_to_plot // parts
                for p in range(parts):
                    # Concatenating all batches dataframes of the same part of the total batches
                    batches_pn = pd.concat(
                        run_input_preds[
                            p * num_batches_part : num_batches_part * (p + 1)
                        ],
                        axis=0,
                    ).to_numpy()

                    #### Creating figure
                    # For fraction_batches_to_plot times reconstruct images in each batch
                    for b in range(fraction_batches_to_plot):

                        batch_size = self.runs_params[f"run_{r}"]["batch_size"]
                        # Creating figure to plot reconstructed images
                        fig = plt.figure(figsize=(10, 14))
                        columns = 6
                        rows = batch_size[0] // columns
                        # start batch
                        start_batch = b * batch_size[0]
                        end_batch = start_batch + batch_size[0]
                        # Slicing arrays corresponding to images from batch b
                        images_this_batch = batches_pn[start_batch:end_batch, :]
                        # ax enables access to manipulate each of subplots
                        ax_images = []

                        for i in range(columns * rows):
                            # Taking batch b in batches_np
                            img = np.reshape(images_this_batch[i, 1:], (28, 28)) * 255
                            im_label = images_this_batch[i, 0]
                            # create subplot and append to ax
                            ax_images.append(fig.add_subplot(rows, columns, i + 1))
                            ax_images[-1].set_title(
                                "Label: " + str(im_label)
                            )  # set title
                            plt.imshow(img)
                        # Save figure
                        plt.savefig(
                            save_dir + "/" + f"run{r}_imgs_{run_type}_p{p}b{b}.pdf",
                            format="pdf",
                            bbox_inches="tight",
                            pad_inches=0.1,
                        )

                # print(f"Stop here {a}")

            elif run_type == "test":
                # Selecting the batches only for current run
                run_input_preds = input_preds[ts_exclude_batches:]
                # Concatenating all batches dataframes saved during testing
                batches_pn = pd.concat(run_input_preds, axis=0).to_numpy()
                # print(batches_pn[0:10, 0:9])
                # print("Fermo qua")
                # Looping over number of batches ot plot
                for b in range(batches_to_plot):
                    batch_size = self.runs_params[f"run_{r}"]["batch_size"]
                    # Creating figure to plot reconstructed images
                    fig = plt.figure(figsize=(10, 14))
                    columns = 6
                    rows = batch_size[1] // columns
                    # start batch
                    start_batch = b * batch_size[1]
                    print(f"Start batch is {b} times {batch_size} equal {start_batch}")
                    end_batch = start_batch + batch_size[1]
                    # Slicing arrays corresponding to images from batch b
                    images_this_batch = batches_pn[start_batch:end_batch, :]
                    # ax enables access to manipulate each of subplots
                    ax_images = []

                    for i in range(columns * rows):
                        # Taking batch b in batches_np
                        img = np.reshape(images_this_batch[i, 1:], (28, 28)) * 255
                        im_label = images_this_batch[i, 0]
                        # create subplot and append to ax
                        ax_images.append(fig.add_subplot(rows, columns, i + 1))
                        ax_images[-1].set_title("Label: " + str(im_label))  # set title
                        plt.imshow(img)
                    # Save figure
                    plt.savefig(
                        save_dir + "/" + f"run{r}_imgs_{run_type}_b{b}.pdf",
                        format="pdf",
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )

    def plt_input_preds_w_transv(self, input_preds, run_type, batch_size, save_dir):
        """
        Function to plot predictions to bottom layer either from training or testing AND producing transversals.

        Inputs:

        - input_preds (list): list of input predictions dataframes (one for every batch for every run);
        - run_type (string): data origin type, 'train' or 'test'
        - exp_id (list): list containing tuples of IDs, every tuple stores the id and its value
          for one experiment;
        - batch_size (list): list with training and testing batch sizes
        - save_dir (string): directory where to save the plots
        """

        # Hard-coded number of batches for which to plot input predictions (images)
        batches_to_plot = 10
        # Retrieving number of runs
        num_runs = len(self.runs)
        # Retrieving number of batches for each run
        runs_num_batches = [
            self.runs_params[f"run_{r}"]["num_batches"] for r in range(num_runs)
        ]
        # Looping over the runs
        for r in range(num_runs):
            # Retrieving value of num_batches parameter
            tr_num_batches = runs_num_batches[r][0]
            ts_num_batches = runs_num_batches[r][1]
            # Number of batches from previous runs
            tr_exclude_batches = sum(
                [runs_num_batches[prev_r][0] for prev_r in range(r)]
            )
            ts_exclude_batches = sum(
                [runs_num_batches[prev_r][1] for prev_r in range(r)]
            )
            # Plotting train top unit activity for run r
            if run_type == "train":
                # Selecting the batches only for current run
                run_input_preds = input_preds[tr_exclude_batches:]
                # Looping over parts of the total bacth number to see some progression (in training)
                parts = 4
                num_batches_part = tr_num_batches // parts
                fraction_batches_to_plot = batches_to_plot // parts
                for p in range(parts):
                    # Concatenating all batches dataframes of the same part of the total batches
                    batches_pn = pd.concat(
                        run_input_preds[
                            p * num_batches_part : num_batches_part * (p + 1)
                        ],
                        axis=0,
                    ).to_numpy()

                    #### Creating figure
                    # For fraction_batches_to_plot times reconstruct images in each batch
                    for b in range(fraction_batches_to_plot):
                        num_transversals = self.runs_params[f"run_{r}"][
                            "num_transversals"
                        ]
                        batch_size = self.runs_params[f"run_{r}"]["batch_size"]
                        # Creating figure to plot reconstructed images
                        fig = plt.figure(figsize=(10, 14))
                        columns = num_transversals
                        rows = batch_size[0] // num_transversals
                        # start batch
                        start_batch = b * batch_size[0]
                        end_batch = start_batch + batch_size[0]
                        # Slicing arrays corresponding to images from batch b
                        images_this_batch = batches_pn[start_batch:end_batch, :]
                        # ax enables access to manipulate each of subplots
                        ax_images = []

                        for i in range(columns * rows):
                            # Taking batch b in batches_np
                            img = np.reshape(images_this_batch[i, 1:], (28, 28)) * 255
                            im_label = images_this_batch[i, 0]
                            # create subplot and append to ax
                            ax_images.append(fig.add_subplot(rows, columns, i + 1))
                            ax_images[-1].set_title(
                                "Label: " + str(im_label)
                            )  # set title
                            # plt.imshow(img)
                        # Save figure
                        plt.savefig(
                            save_dir + "/" + f"run{r}_imgs_{run_type}_p{p}b{b}.pdf",
                            format="pdf",
                            bbox_inches="tight",
                            pad_inches=0.1,
                        )

            elif run_type == "test":
                # Selecting the batches only for current run
                run_input_preds = input_preds[ts_exclude_batches:]
                # Concatenating all batches dataframes saved during testing
                batches_pn = pd.concat(run_input_preds, axis=0).to_numpy()
                print(f"Size of batches_pn: {batches_pn.shape}")
                print(batches_pn[0:10, 0:9])
                print(batches_pn[10:20, 0:9])
                print("Fermo qua")
                # Looping over number of batches ot plot
                for b in range(batches_to_plot):
                    num_transversals = self.runs_params[f"run_{r}"]["num_transversals"]
                    batch_size = self.runs_params[f"run_{r}"]["batch_size"]
                    # Creating figure to plot reconstructed images
                    fig = plt.figure(figsize=(10, 14))
                    columns = num_transversals + 1
                    rows = batch_size[1] // (num_transversals + 1)
                    # start batch
                    start_batch = b * batch_size[1]
                    print(f"Start batch is {b} times {batch_size} equal {start_batch}")
                    end_batch = start_batch + batch_size[1]
                    # Slicing arrays corresponding to images from batch b
                    images_this_batch = batches_pn[start_batch:end_batch, :]
                    # ax enables access to manipulate each of subplots
                    ax_images = []

                    for i in range(columns * rows):
                        # Taking batch b in batches_np
                        img = np.reshape(images_this_batch[i, 1:], (28, 28)) * 255
                        im_label = images_this_batch[i, 0]
                        # create subplot and append to ax
                        ax_images.append(fig.add_subplot(rows, columns, i + 1))
                        ax_images[-1].set_title("Label: " + str(im_label))  # set title
                        plt.imshow(img)
                    # Save figure
                    plt.savefig(
                        save_dir + "/" + f"run{r}_imgs_{run_type}_b{b}.pdf",
                        format="pdf",
                        bbox_inches="tight",
                        pad_inches=0.1,
                    )

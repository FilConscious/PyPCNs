"""
Script to invoke plotting functions to create and save plots.

Created on 27/12/2022 at 17:12:00
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import argparse
import numpy as np
import pandas as pd
from glob import glob

# Custom packages/modules imports
from utilities import plotfs_sup
from utilities import plotfs_unsp


def vis():

    # Parsing command line
    parser = argparse.ArgumentParser()
    # Argument for experiments ID (to be used in plots as labels)
    parser.add_argument(
        "--exp_id",
        "-eid",
        type=str,
        default="exp",
        help="choices: exp, seed, learning_rate, integration_step, last_inf, num_iters",
    )
    # Visualization type (compare, average, comp_avg)
    parser.add_argument("--vis_type", "-vt", type=str, default="compare")
    # Runs type (train, test, trts)
    parser.add_argument(
        "--run_type", "-rt", type=str, default="trts", help="choices: train, test, trts"
    )
    # Network mode
    parser.add_argument("--mode", "-md", type=str, default="supervised")

    ### Parameters Dictionary ###

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Convert args to dictionary
    params = vars(args)
    # Assigning the experiment ID passed via the CL and to be used below
    exp_id = params["exp_id"]
    # Visualization type
    vis_type = params["vis_type"]
    # Run types
    runs_type = params["run_type"]
    # PCN training/testing mode
    pcn_mode = params["mode"]

    ### Dataframes Retrieval and Plotting ###

    # Retrieving directories where results have been stored
    res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
    print(" ")
    print(f"The results directory is: {res_dir}")
    # List of directories with results from different experiment/runs
    # Note: here we are picking only directories that start with 'exp', indicating that they are
    # associated with an experiment
    runs = [t for t in glob(res_dir + "/exp*", recursive=False)]
    # If there are no result directories raise exception
    if len(runs) == 0:

        raise Exception(
            "Sorry, there are no results to visualize. Run an experiment first!"
        )
    # Printing directories of the runs
    print(" ")
    print("Available runs:")
    for i, e in enumerate(runs):
        print(f"- Run {i}: {e}")

    # Creating directory where to store figures
    figures_dir = res_dir + "/figures"
    if not (os.path.exists(figures_dir)):
        os.makedirs(figures_dir)

    # Create custom plotter object
    if pcn_mode == "supervised":
        plotter = plotfs_sup.SupPlotter(runs, runs_type, exp_id, vis_type, figures_dir)
    elif pcn_mode == "unsupervised":
        plotter = plotfs_unsp.UnsPlotter(runs, runs_type, exp_id, vis_type, figures_dir)
    else:
        raise NotImplementedError

    # Using plotter function to load and plot appropriately grouped metrics
    plotter.plot_metrics(runs)


if __name__ == "__main__":
    vis()

"""
Main file for parameters and directories setup.

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Custom packages/modules imports
from scripts.tsk_sup import run

# Note: the above absolute import could be replaced by the *relative import* 'from .run import run_task' with,
# the advantage of not having to write run.run_task(...) when using the function run_task from module run.py.
print("In module tsk_sup sys.path[0], __package__ ==", sys.path[0], __package__)


def main():

    ###### 0. PARSING COMMAND LINE ######

    parser = argparse.ArgumentParser()

    ### Experiment info ###
    # Parameters' configuration file
    parser.add_argument(
        "--config_file",
        "-cfg",
        type=str,
        default=None,
        help="choices: exp0, exp1, exp2, exp3",
    )
    # dataset name
    parser.add_argument(
        "--dataset_name",
        "-dsn",
        type=str,
        default="mnist",
        help="choices: MNIST, dSprites",
    )
    # batch size (training and test)
    parser.add_argument("--batch_size", "-bs", nargs="+", type=int, default=[64, 64])
    # number of batches (training and test)
    parser.add_argument("--num_batches", "-nbs", nargs="+", type=int, default=None)
    # dimensionality of the datapoints
    parser.add_argument("--dim_datapts", "-dimp", type=int, default=784)
    # dimensionality of target
    parser.add_argument("--dim_target", "-dimt", type=int, default=0)
    # no. of iterations (epochs)
    parser.add_argument("--num_iters", "-nitr", type=int, default=1)
    # Setting the random seed
    parser.add_argument("--seed", "-sd", type=float, default=3)

    ### Network info ###
    # no. and sizes of layers
    parser.add_argument("--layers", "-ly", nargs="+", type=int, default=[784, 1024, 10])
    # activation function for the PCN
    parser.add_argument(
        "--activation_function",
        "-af",
        type=str,
        default="sigmoid",
        help="choices: sigmoid, relu, linear",
    )
    # max no. of inference steps for the PCN to reach equilibrium when fed with one datapoint
    parser.add_argument("--inf_cycles", "-ic", nargs="+", type=int, default=[50, 200])
    # epsilon parameter for threshold during inference
    parser.add_argument("--epsilon", "-eps", type=float, default=0.001)
    # learning rate for predictions weights updates
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.0349586720753856
    )
    # learning rate for covariances weights updates
    parser.add_argument(
        "--lr_sigmas",
        "-lrs",
        nargs="+",
        type=float,
        default=[0.07, 0.00000001, 0.00000001],
    )
    # learning rate for top priors updates
    parser.add_argument("--lr_priors", "-lrp", type=float, default=0.01)
    # integration step
    parser.add_argument("--integration_step", "-dt", type=float, default=0.01)
    # Value to initialize top covariance matrix or variance
    parser.add_argument("--top_sigma", "-tsig", type=float, default=0.15)
    # Fixed or learnable top priors (action='store_true' means that IF the flag
    # --top_priors is present the corresponding value is True, otherwise False)
    parser.add_argument("--top_priors", "-tpr", action="store_true")
    # Fixed or learnable covariances
    parser.add_argument("--learn_sigmas", "-lcov", action="store_true")
    # Fixed input activity
    parser.add_argument("--fixed_input_activity", "-fia", action="store_false")
    # Network mode
    parser.add_argument("--mode", "-md", type=str, default="unsupervised")
    # In supervised mode: label at the top or bottom?
    parser.add_argument("--labels_position", "-lbp", type=str, default="None")
    # Num transversals, for testing only
    parser.add_argument("--num_transversals", "-num_trsl", type=int, default=0)
    # Set units for visualizing transversals, for testing only
    parser.add_argument("--set_units", "-su", type=int, default=0)
    # Run process(es)
    parser.add_argument(
        "--process", "-prs", type=str, default="trts", help="choices: tr, ts, trts"
    )
    # Experiment name from which to load saved weights (this is for running further tests only)
    parser.add_argument("--exp_name", "-en", type=str, default="none")

    # Creating object holding the attributes from the command line
    args = parser.parse_args()
    # Datetime object containing current date and time
    now = datetime.now()
    # Converting data-time in an appropriate string: '_dd.mm.YYYY_H.M.S'
    dt_string = now.strftime("_%d.%m.%Y_%H.%M.%S")

    ###### 1. DIRECTORIES ######

    # main.py file directory
    # curr_dir = Path.cwd()
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # Dataset directory
    data_dir = curr_dir + "/datasets"
    if not (os.path.exists(data_dir)):
        os.makedirs(data_dir)

    # Results directory
    res_dir = curr_dir + "/results"
    if not (os.path.exists(res_dir)):
        os.makedirs(res_dir)

    ###### 2. PARAMETERS DICTIONARY  ######

    # Convert args to dictionary
    params = vars(args)

    # Setting experiment name
    exp_name = (
        f'exp{dt_string}DS{params["dataset_name"]}AF{params["activation_function"]}'
        + f'LIF{params["inf_cycles"][0]}LR{params["learning_rate"]:.3f}IST{params["integration_step"]}MD{params["mode"]}SD{params["seed"]}'
    )

    # Create directory where to store experiment metrics and parameters' configuration
    exp_dir = res_dir + "/" + exp_name
    os.makedirs(exp_dir)
    os.makedirs(exp_dir + "/train")
    os.makedirs(exp_dir + "/test")

    if params["exp_name"] != "none":
        net_weights_dir = res_dir + "/" + params["exp_name"]
    else:
        net_weights_dir = None

    # Adding keys/values for relevant defined directories
    params["data_dir"] = data_dir
    params["exp_dir"] = exp_dir
    params["exp_name"] = exp_name
    params["net_weights_dir"] = net_weights_dir

    # Saving configuration file
    with open(exp_dir + "/config.json", "w") as wfile:
        json.dump(params, wfile)

    # Start the experiment with current parameters
    run.run_task(params)


if __name__ == "__main__":
    main()

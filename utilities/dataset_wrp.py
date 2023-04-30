"""
Custom data generation and loading with a wrapper to Pytorch dataloader class..

Created on 07/01/2021 at 15:10:00
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
import pandas as pd
from glob import glob
from numpy.random import default_rng

# Pytorch libraries imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class DataRetrieval(object):
    """Class wrapping Pytorch dataloading objects."""

    datasets_available = ["CIFAR10", "FashionMNIST", "MNIST", "Omniglot"]

    def __init__(self, dataset_name, data_dir, **kwargs):
        """
        Arguments:

        - dataset_name (string): type/name of the dataset to be generated;
        - data_dir (string): directory where to save or from which to load the dataset.
        - num_batches (integer): number of batches
        - batch_size (list): list of two integers for training and testing batch size;
        - set_units (list): list with the index of units for which we will produce transversals
        - num_transversal (integer): number of values to assign to the transversed units
        """

        self.dataset = dataset_name
        self.data_dir = data_dir
        self.kwargs = kwargs

    def retrieve(self):
        """ """

        if self.dataset in self.datasets_available:

            train_ds = eval(
                f"datasets.{self.dataset}(root=self.data_dir, train=True, download=True, transform=ToTensor())"
            )
            test_ds = eval(
                f"datasets.{self.dataset}(root=self.data_dir, train=False, download=True, transform=ToTensor())"
            )

            return train_ds, test_ds

        elif self.dataset == "transversals":

            transversed_ds = TrasvlDataset(
                self.data_dir,
                self.kwargs["num_batches"],
                self.kwargs["batch_size"],
                self.kwargs["set_units"],
                self.kwargs["num_transversals"],
            )

            return None, transversed_ds

        elif self.dataset == "reconstrs":

            reconstrs_ds = ReconstrsDataset(
                self.data_dir,
                self.kwargs["num_batches"],
                self.kwargs["batch_size"],
                self.kwargs["set_units"],
            )

            return None, reconstrs_ds

        else:

            raise NameError("Sorry, the chosen dataset is not available.")


class TrasvlDataset(Dataset):
    """Custom Pytorch dataset from data saved in pandas dataframes used specifically for visualizing
    trasversal after unsupervised learning. The dataset contains top-level priors activity and corresponding
    labels."""

    rng = default_rng()

    def __init__(self, data_dir, num_batches, batch_size, set_units, num_trasversals):
        """
        Arguments:

        - data_dir (string): path to folder with data saved in csv files
        - num_batches (integer): number of saved batches to use for the new dataset
        - set_units (list): list with the index of units that are gradually modified
        - num_trasversal (integer): number of trasversal to generate

        N.B. The code below assumes that the first column in the dataframes
        stores the label of the datapoint that was being processed when the
        data in the remaining column was generated (and then stored).
        """

        # Number of batches in the original experiment (assumed to be a testing one)
        # self.num_batches = num_batches
        # Batch size of every batch in the original experiment (assumed to be a testing one)
        self.bs = batch_size
        # Number of batches for which to compute and visualize transversals
        self.nbt = num_batches
        # Fixed units
        print(f"The list of units is: {set_units}")
        self.set_units = set_units
        # Number of transversals
        self.num_transversal = num_trasversals
        # Using glob to retrieve all the csv files (one csv file contains the data for one batch)
        self.csv_list = [
            t
            for t in glob(
                data_dir + "/transversals/top_units/values_b*", recursive=False
            )
        ]
        # Taking just self.num_batches in the list
        self.rand_csv_list = self.rng.choice(self.csv_list, self.nbt, replace=False)
        # Creating the new dataset
        self.data = self._create_ds()

    def _create_ds(self):
        """Function to create a dataset of trasversals starting from top-unit activity collected during
        the test time in an experiment. Specifically: we are taking self.nbt batches of saved top-unit activity
        and duplicating each vector for self.num_transversal.

        Example. If self.nbt = 2 and self.num_transversal = 10, then we are taking two batches of top-unit
        activity from a certain experiment (depending on data_dir). Suppose those batches have 60 top-unit
        activity vectors each, by copying each of those vectors 10 times we get 1200 vectors in total. Every
        group of ten represents a top-unit vector (associated with a certain label) in which some of the
        units will be modified (within a range of values) to see what changes in the reconstructed image
        when the PCN does inference with that (altered) top-unit activity. To see the evolution from the
        orginal reconstruction (without unit alteration), we are actually making groups of 11 so the total
        number of vectors will be 1320.
        """

        augm_dfs = []

        for d in self.rand_csv_list:
            # Read the dataframe from file
            b = pd.read_csv(d, index_col=0)
            # Duplicating every row of the dataframe for self.num_transversal times
            copies = [self.num_transversal + 1 for _ in b.index.values]
            augm_b = b.loc[np.repeat(b.index.values, copies)]
            augm_dfs.append(augm_b)

        # Concatenating all the dataframes and converting them to numpy
        all_data = pd.concat(augm_dfs, axis=0).to_numpy()
        # Creating a mask with the transversed values in each column repeated for self.bs
        # N.B. We are attaching a 1 at the front so that we can visualize the true reconstruction
        value_mask = np.reshape(
            np.concatenate(
                [
                    np.linspace(-2, 2, num=self.num_transversal)
                    for _ in range(1)  # self.bs * self.nbt)
                ]
            ),
            (-1, 1),
        )
        # print(value_mask.shape)
        # Using the mask to set the values of the corresponding units to the transversed values
        # NOTE: self.set_units must be a list so that units far apart can all be set at once
        # all_data[:, self.set_units] = value_mask.squeeze()

        for r in range(1, all_data.shape[0], self.num_transversal + 1):

            all_data[
                r : r + self.num_transversal, self.set_units
            ] = value_mask.squeeze()

        print(f"Size of all data is {all_data.shape}")
        print(f"First few vectors: {all_data[0:5, 0:5]}")
        print(f"Second group: {all_data[11:16, 0:5]}")
        print(f"Third few vectors: {all_data[22:27, 0:5]}")
        print(f"Fourth few vectors: {all_data[33:38, 0:5]}")
        print(f"Fifth few vectors: {all_data[44:49, 0:5]}")
        print(f"Sixth few vectors: {all_data[55:60, 0:5]}")
        print(f"Seventh few vectors: {all_data[66:71, 0:5]}")

        return torch.from_numpy(all_data)

    # def __iter__(self):

    #     # All the data samples, i.e. the rows of self.data
    #     all_samples = [
    #         (self.data[i, 1:], self.data[i, 0])
    #         for i in range(self.num_transversal * self.bs * self.nbt)
    #     ]

    #     return iter(all_samples)

    def __len__(self):
        """ """
        return self.data.size(dim=0)

    def __getitem__(self, idx):
        """ """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Sample of data/observations with corresponding label
        sample = (self.data[idx, 1:], self.data[idx, 0])

        return sample


class ReconstrsDataset(Dataset):
    """Custom Pytorch dataset from data saved in pandas dataframes used specifically for visualizing
    reconstructed images after unsupervised training. The dataset contains top-level priors activity and
    corresponding labels."""

    rng = default_rng()

    def __init__(self, data_dir, num_batches, batch_size, set_units):
        """
        Arguments:

        - data_dir (string): path to folder with data saved in csv files
        - num_batches (integer): number of saved batches to use for the new dataset
        - set_units (list): list with the index of units that are potentially modified

        N.B. The code below assumes that the first column in the dataframes
        stores the label of the datapoint that was being processed when the
        data in the remaining column was generated (and then stored).
        """

        # Number of batches in the original experiment (assumed to be a testing one)
        # self.num_batches = num_batches
        # Batch size of every batch in the original experiment (assumed to be a testing one)
        self.bs = batch_size
        # Number of batches for which to compute and visualize transversals
        self.nbt = num_batches
        # Fixed units
        print(f"The list of units is: {set_units}")
        self.set_units = set_units
        # Using glob to retrieve all the csv files (one csv file contains the data for one batch)
        self.csv_list = [
            t
            for t in glob(
                data_dir + "/transversals/top_units/values_b*", recursive=False
            )
        ]
        # Taking just self.num_batches in the list
        self.rand_csv_list = self.rng.choice(self.csv_list, self.nbt, replace=False)
        # Creating the new dataset
        self.data = self._create_ds()

    def _create_ds(self):
        """Function to create a dataset of trasversals starting from top-unit activity collected during
        the test time in an experiment. Specifically: we are taking self.nbt batches of saved top-unit activity
        and duplicating each vector for self.num_transversal.

        Example. If self.nbt = 2 and self.num_transversal = 10, then we are taking two batches of top-unit
        activity from a certain experiment (depending on data_dir). Suppose those batches have 60 top-unit
        activity vectors each, by copying each of those vectors 10 times we get 1200 vectors in total. Every
        group of ten represents a top-unit vector (associated with a certain label) in which some of the
        units will be modified (within a range of values) to see what changes in the reconstructed image
        when the PCN does inference with that (altered) top-unit activity. To see the evolution from the
        orginal reconstruction (without unit alteration), we are actually making groups of 11 so the total
        number of vectors will be 1320.
        """

        augm_dfs = []

        for d in self.rand_csv_list:
            # Read the dataframe from file
            b = pd.read_csv(d, index_col=0)
            # Appending the read dataframe to list
            augm_dfs.append(b)

        # Concatenating all the dataframes and converting them to numpy
        all_data = pd.concat(augm_dfs, axis=0).to_numpy()
        # Creating a mask with the transversed values in each column repeated for self.bs
        # N.B. We are attaching a 1 at the front so that we can visualize the true reconstruction
        # value_mask = np.reshape(
        #     np.concatenate(
        #         [
        #             np.linspace(-2, 2, num=self.num_transversal)
        #             for _ in range(1)  # self.bs * self.nbt)
        #         ]
        #     ),
        #     (-1, 1),
        # )
        # print(value_mask.shape)
        # Using the mask to set the values of the corresponding units to the transversed values
        # NOTE: self.set_units must be a list so that units far apart can all be set at once
        # all_data[:, self.set_units] = value_mask.squeeze()

        # for r in range(1, all_data.shape[0], self.num_transversal + 1):

        #     all_data[
        #         r : r + self.num_transversal, self.set_units
        #     ] = value_mask.squeeze()

        # print(f"Size of all data is {all_data.shape}")
        # print(f"First few vectors: {all_data[0:5, 0:5]}")
        # print(f"Second group: {all_data[11:16, 0:5]}")
        # print(f"Third few vectors: {all_data[22:27, 0:5]}")
        # print(f"Fourth few vectors: {all_data[33:38, 0:5]}")
        # print(f"Fifth few vectors: {all_data[44:49, 0:5]}")
        # print(f"Sixth few vectors: {all_data[55:60, 0:5]}")
        # print(f"Seventh few vectors: {all_data[66:71, 0:5]}")

        return torch.from_numpy(all_data)

    def __len__(self):
        """ """
        return self.data.size(dim=0)

    def __getitem__(self, idx):
        """ """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Sample of data/observations with corresponding label
        sample = (self.data[idx, 1:], self.data[idx, 0])

        return sample

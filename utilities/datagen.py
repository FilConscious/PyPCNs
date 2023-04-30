"""
Script to generate different datasets.

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import os
import numpy as np


class DataGen_Mechanisms(object):
    """
    Types of datasets:

    - ab:
    - xcsy:
    - lin_regr:

    """

    def __init__(
        self, dataset_name, num_datapts, dim_datapts, batch_size, save_dir, rng
    ):
        """
        Init method. Arguments:

        - dataset_name (string): type/name of the dataset to be generated;
        - num_datapts (list): list of two integers for numbers of training and testing datapoints;
        - dim_datapts (integer): dimensionality of the datapoints;
        - batch_size (list): list of two integers for training and testing batch size;
        - save_dir (string): directory where to save the dataset;
        - rng: random number generator to sample from probability distributions.

        N.B. self.params is an attribute that will store the parameters used by the generative
        mechanisms to produce data (e.g. means/variances of Gaussians), the parameters are passed
        to the object when the test_data and train_data functions are called.
        """

        self.name_ds = dataset_name
        self.num_tr_dp = num_datapts[0]
        self.num_ts_dp = num_datapts[1]
        self.dim_dpts = dim_datapts
        self.train_bs = batch_size[0]
        self.test_bs = batch_size[1]
        self.save_dir = save_dir
        self.params = {}
        self.rng = rng

    def sample_mean(self, dp):
        """
        Method to compute the sample mean of set of datapoints.

        Inputs:

        - dp (array): datapoints.

        """

        # Note: dp is assumed to be a one-dimensional numpy array
        s_mean = np.sum(dp) / dp.shape[0]

        return s_mean

    def sample_var(self, s_mean, dp):
        """
        Method to compute the sample variance of a set of datapoints.

        Inputs:

        - s_mean (float): sample of mean of the datapoints;
        - dp (array): datapoints.

        """

        # Number of datapoints
        n = dp.shape[0]
        s_variance = (1 / (n - 1)) * np.sum((dp - s_mean) ** 2)

        return s_variance

    def single_rv(self, n, d):
        """
        Function to create data.

        Mechanism:

        - Y = X + n (X -> Y);
        - X is a hidden, fixed cause and Y is the observation the network will receive;
        - n is the (Gaussian) noise that will corrupt the observation.

        N.B. The function will look for the parameters it needs in self.params,
        you need to make sure that the parameter passed are in the right form
        (see below).

        """

        # Retrieving mean/std for a Gaussian, it is assumed that those parameters
        # are stored in a tuple, e.g. (m, s), inside the tuple self.params.
        mean_g = self.params[0][0]
        std_g = self.params[0][1]

        # Retrieving theta
        theta = self.params[1]
        print(f"Theta is {theta}")

        # Data generation for a single random variable (with right prob. ass.)
        # x = self.rng.normal(loc=1.0, scale=2.0, size=(n, d))
        x = self.rng.uniform(5.0, 5.0, size=(n, d))
        y = theta * x + self.rng.normal(loc=mean_g, scale=std_g, size=(n, d))

        return y

    def ab_mechanism(self, n, d):
        """
        Function to create data.

        Mechanisms:

        - X = a + b (A -> X <- B);
        - Y = a - b (A -> Y <- B);
        - both X and Y are observations the network will receive.

        N.B. The function will look for the parameters it needs in self.params,
        you need to make sure that the parameter passed are in the right form
        (see below).
        """

        # Retrieving mean/std for Gaussians, it is assumed that those parameters
        # are stored in a tuple, e.g. (m, s), inside the tuple self.params.
        mean_a = self.params[0][0]
        std_a = self.params[0][1]

        mean_b = self.params[1][0]
        std_b = self.params[1][1]

        # Data generation as in Whittinghton & Bogacz, 2017
        a = self.rng.normal(loc=mean_a, scale=std_a, size=(n, d))
        b = self.rng.normal(loc=mean_b, scale=std_b, size=(n, d))
        x = a + b
        y = a - b

        return x, y

    def xcsy_mechanism(self, n, d):
        """
        Function to create data.

        Mechanisms:

        - Y = X + n (X -> Y);
        - X and n are both Gaussian random variables;
        - X is the cause of Y but it is not hidden;
        - both X and Y are observations the network will receive;
        - n is (Gaussian) noise that corrupts the generation of Y from X.

        N.B. The function will look for the parameters it needs in self.params,
        you need to make sure that the parameter passed are in the right form
        (see below).
        """

        # Retrieving mean/std for Gaussians, it is assumed that those parameters
        # are stored in a tuple, e.g. (m, s), inside the tuple self.params.
        mean_x = self.params[0][0]
        std_x = self.params[0][1]

        mean_n = self.params[1][0]
        std_n = self.params[1][1]

        # Retrieving theta
        theta = self.params[2]

        # Data generation for X -> Y
        x = self.rng.normal(loc=mean_x, scale=std_x, size=(n, d))
        y = theta * x + self.rng.normal(loc=mean_n, scale=std_n, size=(n, d))

        return x, y

    def lin_reg_mechanism(self, n, d):
        """
        Function to create data.

        Mechanisms:

        - Y = theta * X + a (X -> Y);
        - X is a uniform random variable;
        - X is the cause of Y but it is not hidden;
        - both X and Y are observations the network will receive;
        - a is (Gaussian) noise that corrupts the generation of Y from X.

        N.B. The function will look for the parameters it needs in self.params,
        you need to make sure that the parameter passed are in the right form
        (see below).
        """

        # Retrieving mean/std for Gaussians, it is assumed that those parameters
        # are stored in a tuple, e.g. (m, s), inside the tuple self.params.
        mean_a = self.params[0][0]
        std_a = self.params[0][1]

        # Retrieving lower and upper limit for the uniform r.v. (X).
        low_x = self.params[1][0]
        upp_x = self.params[1][1]

        # Retrieving theta
        theta = self.params[2]

        # Data generation for linear regression (with right prob. ass.)
        a = self.rng.normal(loc=mean_a, scale=std_a, size=(n, d))
        x = self.rng.uniform(low_x, upp_x, size=(n, d))
        y = theta * x + a

        # Data generation for linear regression (with right prob. ass.)
        # y = self.rng.uniform(-5.0, 25.0, size=(n, d))
        # new_x = y / theta - a / theta

        # return y, new_x

        return x, y

    def gen_process(self, n, d):
        """ """

        # Data generation via "mechanisms" methods
        if self.name_ds == "ab":

            x, y = self.ab_mechanism(n, d)
            concat_inputs = np.concatenate((x, y), axis=1)

        elif self.name_ds == "xcsy":

            x, y = self.xcsy_mechanism(n, d)
            concat_inputs = np.concatenate((x, y), axis=1)

        elif self.name_ds == "lin_regr":

            x, y = self.lin_reg_mechanism(n, d)
            concat_inputs = np.concatenate((x, y), axis=1)

        elif self.name_ds == "single_rv":

            x = self.single_rv(n, d)
            concat_inputs = x

        else:

            raise NameValueError("Invalid dataset name.")

        return concat_inputs

    def train_data(self, *params):
        """
        Function to prepare and return the training dataset.

        Inputs:

        - params (tuple): tuple with a variable number of parameters for the generative mechanisms.

        Output:

        - batched_data (array): numpy array of shape (b, n, d) where b is the number
        of batches, n is the number of datapoints in one batch, and d is the dimension
        of each datapoint.
        """

        # Setting the parameter attribute
        self.params = params

        if os.path.exists(self.save_dir + f"/{self.name_ds}-train.npz"):

            print("Training data has already been created, loading saved dataset...")

            with np.load(self.save_dir + f"/{self.name_ds}-train.npz") as load:
                batched_data = load["bd"]

            # Printing some statistics about training data
            print("Printing some statistics about training data")
            print(f"Shape of data: {batched_data.shape}")

            # Distinction between Task 1 (statistics for one observation only) and Task 2 (for two observations)
            if batched_data.shape[-1] > 1:

                sm_X = self.sample_mean(batched_data.squeeze()[:, 0])
                sm_Y = self.sample_mean(batched_data.squeeze()[:, 1])
                print(f"Sample mean of X: {sm_X}")
                print(
                    f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze()[:,0])}"
                )
                print(f"Sample mean of Y: {sm_Y}")
                print(
                    f"Sample variance of Y: {self.sample_var(sm_Y, batched_data.squeeze()[:,1])}"
                )

            else:
                sm_X = self.sample_mean(batched_data.squeeze())
                print(f"Sample mean of X: {sm_X}")
                print(
                    f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
                )

            print("Done.")

            return batched_data

        # Generative process returns input data
        data = self.gen_process(self.num_tr_dp, self.dim_dpts)

        # Determining number of batches
        num_batches = int(self.num_tr_dp / self.train_bs)
        # Arranging and randomizing datapoints (rows of two-dimensional array) into num_batches batches of size batch_size
        batched_data = self.rng.choice(
            data, size=(num_batches, self.train_bs), replace=False, axis=0, shuffle=True
        )

        #### Defensive Programming ####
        # assert batched_data.shape[2] == 1*self.dim_dpts, f"Data does not have the right dimensionality! batched_data.shape[2] is {batched_data.shape[2]}, self.dim_dpts is {self.dim_dpts}"
        #### End of DP ####

        # Saving the dataset as .npz file if not already done
        print("Saving training dataset...")
        np.savez(self.save_dir + f"/{self.name_ds}-train.npz", bd=batched_data)

        # Printing some statistics about training data
        print("Printing some statistics about training data")
        print(f"Shape of data: {batched_data.shape}")

        # Distinction between Task 1 (statistics for one observation only) and Task 2 (for two observations)
        if batched_data.shape[-1] > 1:

            sm_X = self.sample_mean(batched_data.squeeze()[:, 0])
            sm_Y = self.sample_mean(batched_data.squeeze()[:, 1])
            print(f"Sample mean of X: {sm_X}")
            print(
                f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze()[:,0])}"
            )
            print(f"Sample mean of Y: {sm_Y}")
            print(
                f"Sample variance of Y: {self.sample_var(sm_Y, batched_data.squeeze()[:,1])}"
            )

        else:
            sm_X = self.sample_mean(batched_data.squeeze())
            print(f"Sample mean of X: {sm_X}")
            print(
                f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
            )

        print(f"Some training datapoints: {batched_data.squeeze()[0:20]}")

        print("Done.")

        return batched_data

    def test_data(self, *params):
        """
        Function to prepare and return the test dataset.

        Input:

        - params(tuple): tuple with variable number of parameters for the generative mechanisms.

        Output:

        - batched_data (array): numpy array of shape (b, n, d) where b is the number
        of batches, n is the number of datapoints in one batch, and d is the dimension
        of each datapoint.
        """

        # Setting the parameter attribute
        self.params = params

        if os.path.exists(self.save_dir + f"/{self.name_ds}-test.npz"):

            print("Test data has already been created, loading saved dataset...")

            with np.load(self.save_dir + f"/{self.name_ds}-test.npz") as load:
                batched_data = load["bd"]

            # Printing some statistics about testing data
            print("Printing some statistics about testing data")
            print(f"Shape of data: {batched_data.shape}")

            # Distinction between Task 1 (statistics for one observation only) and Task 2 (for two observations)
            if batched_data.shape[-1] > 1:

                sm_X = self.sample_mean(batched_data.squeeze())
                sm_Y = self.sample_mean(batched_data.squeeze()[:, 1])
                print(f"Sample mean of X: {sm_X}")
                print(
                    f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
                )
                print(f"Sample mean of Y: {sm_Y}")
                print(
                    f"Sample variance of Y: {self.sample_var(sm_Y, batched_data.squeeze()[:,1])}"
                )

            else:
                sm_X = self.sample_mean(batched_data.squeeze())
                print(f"Sample mean of X: {sm_X}")
                print(
                    f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
                )

            print("Done.")

            return batched_data

        # Generative process returns input data
        data = self.gen_process(self.num_ts_dp, self.dim_dpts)

        # Determining number of batches
        num_batches = int(self.num_ts_dp / self.test_bs)
        # Arranging and randomizing datapoints (rows of two-dimensional array) into num_batches batches of size batch_size
        batched_data = self.rng.choice(
            data, size=(num_batches, self.test_bs), replace=False, axis=0, shuffle=True
        )

        # Saving the dataset as .npz file if not already done
        print("Saving training dataset...")
        np.savez(self.save_dir + f"/{self.name_ds}-test.npz", bd=batched_data)

        # Printing some statistics about testing data
        print("Printing some statistics about testing data")
        print(f"Shape of data: {batched_data.shape}")

        # Distinction between Task 1 (statistics for one observation only) and Task 2 (for two observations)
        if batched_data.shape[-1] > 1:

            sm_X = self.sample_mean(batched_data.squeeze())
            sm_Y = self.sample_mean(batched_data.squeeze()[:, 1])
            print(f"Sample mean of X: {sm_X}")
            print(
                f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
            )
            print(f"Sample mean of Y: {sm_Y}")
            print(
                f"Sample variance of Y: {self.sample_var(sm_Y, batched_data.squeeze()[:,1])}"
            )

        else:
            sm_X = self.sample_mean(batched_data.squeeze())
            print(f"Sample mean of X: {sm_X}")
            print(
                f"Sample variance of X: {self.sample_var(sm_X, batched_data.squeeze())}"
            )

        print(f"Some testing datapoints: {batched_data.squeeze()[0:20]}")

        print("Done.")

        return batched_data


class BatchDataSup:
    """ """

    def __init__(self, dataset, labels, batch_size, num_batches, rng):

        self.datapoints = dataset
        self.labels = labels
        self.batch_size = batch_size
        self.num_datapoints = self.datapoints.shape[0]
        self.curr_indices = list(np.arange(self.num_datapoints))
        # self.curr_indices = list(np.arange(640))
        # Either using num_batches from user or setting based on batch_size and number of datapoints
        if num_batches != None:
            self.num_batches = num_batches
        else:
            self.num_batches = len(self.curr_indices) // self.batch_size
        # Numpy random number generator
        self.rng = rng

        self.batches = []

    def batch_gen(self):

        curr_batch = []

        for _ in range(self.batch_size):

            curr_index = self.rng.choice(
                self.curr_indices, size=1, replace=False, shuffle=True
            )
            curr_datapoint = self.datapoints[curr_index] / 255.0
            curr_label = self.labels[curr_index]
            # Removing from the list the index just drawn, effectively manually implementing a draw
            # without replacement (so replace=False above is not necessary), this is required so that the
            # length of the indices list decreases and can be used to stop the batch formation (see if
            # condition in __next__ below)
            self.curr_indices.remove(curr_index)
            curr_batch.append((curr_datapoint, curr_label))

        self.batches.append(curr_batch)

        return curr_batch

    def __iter__(self):

        return self

    def __next__(self):

        if len(self.batches) < self.num_batches:
            return self.batch_gen()
        else:
            print(f"No more batches of size {self.batch_size} can be created")
            raise StopIteration

    def print_num_batches(self):

        return self.num_batches

    def reset_index_list(self):

        self.curr_indices = list(np.arange(self.num_datapoints))

        print("Index list reset, ready to start a new iteration.")


def PrepMNIST(data_dir, batch_size, rng, num_batches=None):
    """ """

    tr_bs = batch_size[0]
    ts_bs = batch_size[1]

    with np.load(data_dir) as load:

        train_examples = load["x_train"]
        train_labels = load["y_train"]
        test_examples = load["x_test"]
        test_labels = load["y_test"]

    batched_train_data = BatchDataSup(
        train_examples, train_labels, tr_bs, num_batches, rng
    )
    batched_test_data = BatchDataSup(
        test_examples, test_labels, ts_bs, num_batches, rng
    )
    iter_train_data = iter(batched_train_data)
    iter_test_data = iter(batched_test_data)

    return iter_train_data, iter_test_data

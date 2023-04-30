"""
Module defining the logger class.

Created on Sat Mar 05 11:55:00 2022
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np


class logger(object):
    """
    Logger object class to store metrics from experiments with a PCN.

    There are two types of experiments characterized by different streams of data:

    1. one at a time all the datapoints (e.g. x-y pairs) are feed to the PCN, an iteration ends when all
    the datapoints have been processed, and a new iteration may start with the same datapoints;
    2. the datapoints are divided into batches (e.g. a batch with 60 images), an epoch consists in the
    processing of one batch, an iteration is the complete processing of all the batches

    PCN data logged from type-1 experiments are at the datapoint level (e.g. storing the predictions for every
    single (x-y pairs)) whereas data logged from type-2 experiments is usually at the batch level (e.g. either
    average batch predictions or a single prediction from a datapoint in the batch).

    """

    def __init__(self, num_iters, num_dpts, batch_size):
        """
        Init method. Arguments:

        - num_iters (integer): number of training iterations (passes over the entire dataset), more than one
        for type-1 exps, usually one for type-2 exps;
        - num_dpts (tuple): number of datapoints (type-1 exps) or batches (type-2 exps) per training and
        testing iteration;
        - batch_size (list): numbers of datapoints in a batch for training and testing, this is equal to one in
        type-1 exps, more than one in type-2 exps.
        """

        # Data info
        self.num_it = num_iters
        self.num_dpt = num_dpts
        self.bs = batch_size

        # Attributes where to store experiment metrics (for training and testing)
        # N.B. All the list below will be of length equal to self.num_it

        # Last free energies for each iteration (lists of lists)
        # To clear: self.last_ftra will store lists of length equal to self.num_dpt * self.bs,
        # e.g. in type-1 exps there will be self.num_it lists of length self.num_dpt * 1, in type-2 exps
        # usually there will be *one* list of length self.num_dpt * self.bs.
        self.last_ftra = []
        self.last_ftes = []
        # Labels for each iteration (lists of lists)
        self.all_labels_tr = []
        self.all_labels_ts = []
        # Errors at bottom layer for each iteration (lists of lists)
        self.tr_errs = []
        self.ts_errs = []
        # RMSE for each iteration
        self.tr_rmse = []
        self.ts_rmse = []
        # Last inference predictions for every datapoint
        self.tr_preds = []
        self.ts_preds = []
        # Input layer activity
        self.ts_inl = []
        # Last inference top layer activity
        self.tr_tpu = []
        self.tr_tau = []
        # Last inference top layer activations
        self.ts_tpu = []
        self.ts_tau = []
        # Inferential steps
        self.inf_counts = []
        # Covariances matrices
        self.all_cov_matrices = []
        # Weight matrices
        self.all_w_matrices = []
        # Priors on the top layer
        self.all_tl_priors = []

        # Attributes where to store *temporary* experiment metrics:
        # N.B. The list below will be of length equal to self.num_dpt * self.bs

        # Last free energy for each datapoint in all batches
        self.last_fes = []
        # Labels/targets in the order seen by the network
        self.labls = []
        # Error(s) at the bottom layer for each datapoint
        self.lb_errs = []
        # Predictions directed to bottom layer at last inference
        self.last_preds = []
        # Input layer activity at last inference (only for testing)
        self.input_layer = []
        # Activities and activations at the top layer for each datapoint
        self.last_tpu = []
        self.last_tau = []
        # Inferential steps for each datapoint in current iteration
        self.curr_inf_count = []
        # Covariances matrices for current iteration
        self.cov_matrices = []
        # Weight matrices for current iteration
        self.w_matrices = []
        # Priors on the top layer for current iteration
        self.tl_priors = []

    def log_lfe(self, last_fe, test=False):
        """
        Method to log last free energies. For every datapoint/batch we let the PCN
        reach equilibrium. At that point, we want to save the last computed
        total free energy. We do this for every datapoint in a batch for every iteration.

        Inputs:

        - last_fe (float): last free energy;
        - test (Boolean): flag to determine whether we are at test time.
        """

        ### Defensive Programming ###
        # assert (
        #     type(last_fe) == float
        # ), f"Free energy should be of type float but is of type {type(last_fe)}"
        ### End of DP ###

        # Storing last_fe for current batch
        self.last_fes.append(last_fe)
        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.last_fes) == self.num_dpt[0] * self.bs[0] and not (test):

            self.last_ftra.append(self.last_fes)

        elif len(self.last_fes) == self.num_dpt[1] * self.bs[1] and test:

            self.last_ftes.append(self.last_fes)

    def log_labls(self, last_label, test=False):
        """
        Method to log datapoints labels. Only used for supervised tasks (e.g. MNIST).

        Inputs:

        - last_label (integer): label indentifying the datapoint;
        - test (Boolean): flag to determine whether we are at test time.
        """

        ### Defensive Programming ###
        # assert (
        #     last_label.shape == ()
        # ), f"Class label should be a singleton but it has dimension {last_label.shape}."
        ### End of DP ###

        # Storing last_fe for current batch
        self.labls.append(last_label)
        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.labls) == self.num_dpt[0] * self.bs[0] and not (test):

            self.all_labels_tr.append(self.labls)

        elif len(self.labls) == self.num_dpt[1] * self.bs[1] and test:

            self.all_labels_ts.append(self.labls)

    def log_errors(self, errors, test=False):
        """
        Method to log prediction errors at the bottom (input) layer. We do this
        at the last inferential cycle for every datapoint and for each batch.

        Inputs:

        - errors (array): array storing the prediction errors for each error unit
        at the bottom layer for a single datapoint;
        - test (Boolean): flag to determine whether we are at test time.
        """

        ### Defensive Programming ###
        assert (
            type(errors) == np.ndarray
        ), f"Variable errors not of type array, {type(errors)} instead."
        ### End of DP ###

        # Storing errors at bottom layer for current batch
        self.lb_errs.append(errors)
        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.lb_errs) == self.num_dpt[0] * self.bs[0] and not (test):

            self.tr_errs.append(self.lb_errs)

        elif len(self.lb_errs) == self.num_dpt[1] * self.bs[1] and test:

            self.ts_errs.append(self.lb_errs)

    def log_preds(self, p, test=False):
        """
        Method to log prediction for every datapoints during training and testing.

        N.B. In type-2 exps predictions will be stored just for one datapoint for every batch
        so self.last_preds will be a list of length self.num_dpt[0] where the latter is the
        number of batches for training (similarly for testing).
        """

        # Storing single prediction in list for current iteration
        self.last_preds.append(p)
        # When all the datapoints have been processed, store the list in
        # another one that tracks predictions over iterations.

        # print(f'Number of predictions for batch: {len(self.batch_preds)}')
        # print(f'Batch size: {self.bs[1]}')
        if len(self.last_preds) == self.num_dpt[0] * self.bs[0] and not (test):

            print("Saving predictions for current training iteration.")
            self.tr_preds.append(self.last_preds)

        elif len(self.last_preds) == self.num_dpt[1] * self.bs[1] and test:

            print("Saving predictions for current testing iteration.")
            self.ts_preds.append(self.last_preds)

    def log_inl(self, lz):
        """
        Method to log input layer activity for every datapoints during testing.
        """

        # Storing single input activity array in list for current datapoint
        self.input_layer.append(lz)
        # If all the datapoints have been processed, store all data in another list tracking
        # the input layer activity over iterations.
        if len(self.input_layer) == self.num_dpt[1]:

            print("Saving testing input activity.")
            self.ts_inl.append(self.input_layer)

    def log_topl(self, l_tpu, l_tau, test=False):
        """
        Method to log top layer data, i.e. activations (l_tau) and updated activity (l_tpu) at the
        last inference pass, for every datapoint during training.

        N.B. In type-2 exps activity will be stored just for one datapoint for every batch so self.last_tpu
        and self.last_tau will be lists of length self.num_dpt[0] where the latter is the number of batches
        for training (similarly for testing).
        """

        # Storing top layer data in lists for current batch
        self.last_tpu.append(l_tpu)
        self.last_tau.append(l_tau)

        # When all the datapoints have been processed, store all data in another
        # list tracking the top layer data over iterations.
        if len(self.last_tpu) == self.num_dpt[0] * self.bs[0] and not (test):

            print("Saving training l_tpu and l_tau.")
            self.tr_tpu.append(self.last_tpu)
            self.tr_tau.append(self.last_tau)

        elif len(self.last_tpu) == self.num_dpt[1] * self.bs[1] and test:

            print("Saving testing l_tpu and l_tau.")
            self.ts_tpu.append(self.last_tpu)
            self.ts_tau.append(self.last_tau)

    def log_priors(self, priors):
        """
        Method to log top level priors at the end of every batch. So, if there
        are n epochs where at each epoch one batch is processed, there will be
        n saved top level priors.

        Input:

        - priors (array): array storing the top level priors.

        N.B. This function presupposes that at every epoch the PCN processes only
        one batch.
        """

        # Copy the array of top level priors for current iteration
        copy_tlpr = priors.copy()
        # Storing the copied array in list of top level priors
        self.tl_priors.append(copy_tlpr)

        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.tl_priors) == self.num_dpt[0]:

            self.all_tl_priors.append(self.tl_priors)

    def log_cov(self, cov):
        """
        Method to log covariance matrices after every update. In type-1 exps this happens for every
        datapoint, in type-2 task for every batch..

        Input:

        - cov (list): list of arrays representing the covariance matrices of the PCN.

        """

        # List to store a copy of the arrays for current iteration
        copy_sigmas = []

        # Iterating over the arrays and copying them
        for a in cov:

            copy_a = a.copy()
            copy_sigmas.append(copy_a)

        # Storing the list of copied arrays in another list
        self.cov_matrices.append(copy_sigmas)

        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.cov_matrices) == self.num_dpt[0]:

            self.all_cov_matrices.append(self.cov_matrices)

    def log_weights(self, w):
        """
        Method to log weight matrices at the end of every batch. So, if there are n epochs where
        at each epoch one batch is processed, there will be n saved weight matrices.

        Input:

        - w (list): list of arrays representing the weight matrices of the PCN.

        N.B. 1: w is a list that corresponds to the attribute pcn.weights storing
        the weight matrices of the PCN, however note that the first element of
        that list is the float 0.0 which is there just for consistency, indicating
        that layer 0 does not have any weights.
        N.B. 2: This function presupposes that at every epoch the PCN processes only
        one batch.
        """

        # List to store a copy of the arrays for current iteration
        copy_weights = []

        # Iterating over the arrays and copying them
        for m in w:

            if type(m) == float:

                pass

            else:

                # print(type(m))
                # print(m.shape)

                copy_m = m.copy()
                copy_weights.append(copy_m)

        # Storing the list of copied arrays in another list
        self.w_matrices.append(copy_weights)

        # If all the datapoints have been processed, store all data in iterations lists
        if len(self.w_matrices) == self.num_dpt[0]:

            self.all_w_matrices.append(self.w_matrices)

    def log_counter(self, c, test=False):
        """
        Method to log the number of inferences the PCN takes for each datapoint
        over iterations.
        """

        # Storing inference count in list for current iteration
        self.curr_inf_count.append(c)

        # When all the datapoints have been processed, store the counts in another list
        # tracking the inference count data over iterations.
        if len(self.curr_inf_count) == self.num_dpt[0] and not (test):

            print("Saving training inference counts.")
            self.inf_counts.append(self.curr_inf_count)

        elif len(self.curr_inf_count) == self.num_dpt[1] and test:

            raise NotImplementedError()

    # def avg_fe(self):
    #     """
    #     Method to compute average free energy over the iterations.
    #     """

    #     if len(self.tr_fes.keys()) and len(self.ts_fes.keys()) == 1:
    #         # Only one big batch and we train the PCN on that for num_iters.
    #         # Mean last free energy for the train and test batches is appended
    #         # to corresponding lists.
    #         self.avg_ftra.append(np.mean(self.tr_fes[0]))
    #         self.avg_ftes.append(np.mean(self.ts_fes[0]))

    #     elif len(self.tr_fes.keys()) > 1:
    #         # Several batches on which we train the PCN for num_iters.
    #         # Mean last free energy for each batch and its mean over batches
    #         # is appended to corresponding lists. But for testing we use only
    #         # one batch so same line as above.
    #         self.avg_ftra.append(np.mean(np.array(list(self.tr_fes.values()))))
    #         self.avg_ftes.append(np.mean(self.ts_fes[0]))

    #     else:

    #         raise ValueError("Wrong number of keys in free energy dictionaries.")

    # def rmse(self):
    #     """
    #     Method to compute RMSE over the iterations.
    #     """

    #     if len(self.tr_fes.keys()) and len(self.ts_fes.keys()) == 1:
    #         # Only one big batch and we train the PCN on that for num_iters.
    #         # In a way similar to avg_fe(), we compute the RMSE...
    #         r_train = np.sqrt(np.mean(np.array(self.tr_errs[0]).squeeze() ** 2, axis=0))
    #         r_test = np.sqrt(np.mean(np.array(self.ts_errs[0]).squeeze() ** 2, axis=0))
    #         self.tr_rmse.append(r_train)
    #         self.ts_rmse.append(r_test)

    #     elif len(self.tr_fes.keys()) > 1:
    #         # Several batches on which we train the PCN for num_iters.
    #         # In a way similar to avg_fe(), we compute the RMSE... For
    #         # testing we use only one batch so same line as above.
    #         r_train = np.sqrt(
    #             np.mean(
    #                 np.array(list(self.tr_errs.values())).squeeze() ** 2, axis=(0, 1)
    #             )
    #         )
    #         r_test = np.sqrt(np.mean(np.array(self.ts_errs[0]).squeeze() ** 2, axis=0))
    #         self.tr_rmse.append(r_train)
    #         self.ts_rmse.append(r_test)

    # NOTE. When we compute the RMSE in the first case (only one batch), we
    # are taking a list of arrays (e.g. train_errs[0], the first value in the
    # dictionary train_errs) where each array stores the errors from the units
    # in the bottom layer for one datapoint. Those arrays should be of size (n, 1)
    # where n is the number of error units at the bottom layer (same as the
    # dimensionality of the input). By using np.array(), that list of arrays
    # becomes an array of size (m,n,1) where m is the batch size, and now the
    # second dimension corresponds to the error unit. We use squeeze() to remove
    # the singleton dimension. When we apply the np.mean() we are averaging over
    # rows, i.e. over the different datapoints, so that we end up with an array
    # showing the RMSE for each unit for that batch.
    #
    # In the second case (multiple batches), we have to deal with a list of list
    # of arrays. So, from using np.array() and squeezing it, we get a tensor of
    # size (b,m,n) where m and n are defined as before whereas b stands for the
    # batch number. This time we need to apply np.mean() twice to get an RMSE
    # averaged over batches (for each error unit).

    def reset_temp(self, end_reset=False):
        """
        Method to reset temporary lists before starting new iteration.
        """

        if end_reset:

            # Last free energy for each datapoint
            self.last_fes = []
            # Labels for each datapoint
            self.labls = []
            # Error(s) at the bottom layer for each datapoint
            self.lb_errs = []
            # Predictions directed to bottom layer at last inference
            self.last_preds = []
            # Input layer activity at last inference (only for testing)
            self.input_layer = []
            # Activities and activations at the top layer for each datapoint
            self.last_tpu = []
            self.last_tau = []
            # Inferential steps for each datapoint in current iteration
            self.curr_inf_count = []
            # Covariances matrices for current iteration
            self.cov_matrices = []
            # Weight matrices for current iteration
            self.w_matrices = []
            # Priors on the top layer for current iteration
            self.tl_priors = []

            print("Temporary variables have been reset.")

"""
Module defining the predictive coding network class with Pytorch.

Created on Tue Dec 12 2022 at 07:52:00
@author: Filippo Torresan
"""

# Standard libraries imports
import numpy as np
import matplotlib.pyplot as plt

# Pytorch libraries imports
import torch
from torch import nn
from torch import optim
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence

# Global variables
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# PCN class in Pytorch
class TorchPcn(nn.Module):
    """ """

    def __init__(
        self,
        layers,
        activ_func,
        eps,
        batch_size,
        lr,
        lr_sigmas,
        lr_priors,
        dt,
        last_inf,
        target_dim,
        top_priors,
        top_sigma,
        learn_sigmas,
        labels_pos,
        mode="unsupervised",
        fixed_input_activity=True,
        load_model=(False, None),
    ):
        super().__init__()
        """
        Init method. Arguments:

        - layers (list): list of integers, e.g. [1,2,2], where 1 is the size of
        the input layer (input dimensionality) indexed by 0, 2 is
        the size of the first layer (two predictive units and error
        units) indexed by 1, etc.;
        - activ_func: activation function;
        - eps (float): threshold to decide when to stop network inference;
        - batch_size (integer): number of datapoints in a batch;
        - lr (float): learning rate for the update of the "prediction" weights;
        - lr_sigmas (list): list of learning rates for the update of the covariances' weights,
        one for each layer (NOTE: this can be arranged in such a way that the learning actually
        happens in only one layer, i.e. by passing a learning rate close to zero);
        - lr_priors (float): learning rate for the update of the top priors;
        - dt (float): integration step for inference over variables;
        - last_inf (integer): maximum number of inference cycles;
        - target_dim (integer): dimensionality of the part of the input to be predicted at test time;
        - top_priors (boolean): flag for whether or not top layer priors are present or not;
        - top_sigma (float): value to initialize top covariance matrix or top variance;
        - learn_sigmas (boolean): flag for whether or not covariances are learned;
        - mode: flag to run the network either in supervised or unsupervised mode;
        - fixed_input_activity (boolean): flag for having input units that updated during inference.
        - load_model (tuple): the first element is a Boolean indicating whether the network weights will
        be loaded from file, the second element is the path to that file

        (N.B. Now this is implemented and used only for Task 1);
        for Task 1)
        """

        # Initializing parameters
        self.layers = layers
        self.af_str = activ_func
        # self.activ_func = self.set_activ_func(activ_func)
        self.bs = batch_size
        self.lr = lr
        self.lr_sigmas = lr_sigmas
        self.lr_priors = lr_priors
        self.dt = dt
        self.eps = eps
        self.inf_cycles_values = [last_inf[0], last_inf[1]]
        self.last_inf = self.inf_cycles_values[0]
        self.t_dim = target_dim
        self.learn_sigmas = learn_sigmas
        self.top_priors = top_priors
        self.fia = fixed_input_activity
        self.mode = mode
        self.labels_pos = labels_pos
        self.errors = []
        self.error_units = []
        self.activ_units = []
        self.total_batch_fe = torch.zeros(1, 1).to(DEV)
        # List of PCN layers/modules
        self.pcn_layers = nn.ModuleList([])
        # Creating the modules/layers
        self.init_layers()
        # Initializing modules parameters
        self.apply(self.init_params)

        # for parameter in self.parameters():
        # print(f"Size of parameter: {parameter.size()}")
        # print(parameter)
        # print(self.learn_sigmas)

        # for m in self.pcn_layers:
        #     print(m)

    @property
    def last_inf(self):
        """ """
        return self._last_inf

    @last_inf.setter
    def last_inf(self, val):
        """ """
        print(f"Setting maximum number of inferential cycles to {val}.")
        self._last_inf = val

    @torch.no_grad()
    def init_layers(self):
        """ """

        torch.set_default_dtype(torch.float64)

        for idx, s in enumerate(self.layers):

            if idx == 0:
                # Input layer: needs target dimension "from above" and fixed activity required
                p_module = PcnPreds(
                    s,
                    s,
                    self.bs,
                    fix_activity=self.fia,
                    input_layer=True,
                    mode=self.mode,
                )
                e_module = PcnErrs(s, self.bs, self.learn_sigmas)
                e_module.weights = e_module.weights * (1000000)

            elif idx == (len(self.layers) - 1):
                # Top layer
                if self.mode == "unsupervised":
                    # Dummy error units and activity updates in unsupervised mode
                    p_module = PcnPreds(
                        s,
                        self.layers[idx - 1],
                        self.bs,
                        fix_activity=False,
                        mode=self.mode,
                    )
                    e_module = PcnErrs(s, self.bs, False)
                else:
                    # Dummy error units and activity is fixed in supervised mode
                    p_module = PcnPreds(
                        s,
                        self.layers[idx - 1],
                        self.bs,
                        fix_activity=True,
                        mode=self.mode,
                    )
                    e_module = PcnErrs(s, self.bs, False)
            else:
                # Hidden layer: target dimension "from below"
                p_module = PcnPreds(
                    s, self.layers[idx - 1], self.bs, fix_activity=False, mode=self.mode
                )
                e_module = PcnErrs(s, self.bs, self.learn_sigmas)

            self.pcn_layers.append(p_module)
            self.pcn_layers.append(e_module)

    @torch.no_grad()
    def init_params(self, m):
        """
        Method to initialize modules weights etc.. This is called when the PCN is instantiated.

        Arguments:

         - m: torch.nn module
        """

        if isinstance(m, PcnPreds) and m.input_layer == False:

            nn.init.xavier_uniform_(m.weights, gain=nn.init.calculate_gain(self.af_str))
            nn.init.normal_(m.bias)
            # m.bias.fill_(0.0)

        elif isinstance(m, PcnErrs):

            if m.learn_weights == False:
                # Weights (covariance matrix) already fixed when module was created
                pass
            else:
                # Some matrix operations with the random sample to make sure
                # that the covariance matrix is symmetric
                r_m = torch.empty_like(m.weights)
                nn.init.normal_(r_m, mean=0.0, std=1.0)
                id_m = (-1) * torch.eye(r_m.size()) * r_m
                m.weights = id_m + r_m + r_m.T

    @torch.no_grad()
    def init_preds(self, m):
        """
        Method to initialize modules prediction units. This is needed only for modules of class PcnPreds
        and is called at the start of every inference cycle.

        Arguments:

         - m: torch.nn module
        """
        if isinstance(m, PcnPreds) and m.input_layer == False:

            nn.init.uniform_(m.units, 0.0, 1.0)

    def load_pcn(self, load_path):
        """Method to load saved weights of the PCN"""

        raise NotImplementedError

    def select_params(self, inner=False):
        """Method to select which parameters to optimize. This is needed to select and feed
        *only* predictive units activity to the optimizer at the end of every inferential cycle
        (inner optimization), and to select and feed *only* weights/covariance matrices as
        parameters to optimize at the end of every batch."""

        params_to_opt = []

        # Loop over PCN modules
        for m in self.pcn_layers:
            # Inner optimization: select only units params
            if inner:
                # Unit params are updated only for PcnPreds modules
                if isinstance(m, PcnPreds) and m.fix_activity == False:
                    params_to_opt.append(m.units)
            # Outer optimization: select weights and bias
            else:
                # PcnPreds updates weights and bias
                if isinstance(m, PcnPreds) and m.input_layer == False:
                    params_to_opt.append(m.weights)
                    params_to_opt.append(m.bias)
                # PcnErrs optimizes only weights
                elif isinstance(m, PcnErrs):
                    params_to_opt.append(m.weights)

        return params_to_opt

    def forward(self):
        """
        Method to run the network until equilibrium, meaning: predictions flow down
        and are compared to activity at that layer, prediction errors are computed,
        then the errors are used to update the activity at every layer.

        Inputs:

        - None

        Outputs:

        - last_F[-1] (float): the last free energy computed while dealing with a
        single datapoint;
        - last_pred[-1] (array): the last predictions directed to the bottom layer;
        - self.pred_units[0] (array): the activity at the bottom layer, note that
        this is interesting only at test time when the network tries to complete the
        input (at train time the input layer just stores the given input data).

        N.B. Here we are computing everything layer by layer starting from the
        bottom, i.e. starting with layer 1, we compute its predictions arriving
        at layer 0 (input layer), compute the prediction errors there, and update
        the activity of the prediction units at layer 1. Then we move on to layer
        2 and do the same...
        """

        last_F = [5000000.0, 1000000.0]
        last_in_preds = None
        counter = 0

        inner_opt = optim.SGD(self.select_params(inner=True), lr=self.dt)

        # Main inference process
        # NOTE: we are singling all but the *last* inference step in which gradients are required
        # while (abs(last_F[-2] - last_F[-1]) >= self.eps) and (counter != self.last_inf):
        while counter != self.last_inf:
            # print(f"Inference {counter}")
            if (counter + 1) != self.last_inf:

                curr_fe, _ = self.InferCycle(counter)
                # print(f"Old prediction units {self.pred_units[1][0:9]}")
                # print(f"FE on which inference is performed: {tot_fe}")
                # INNER OPTIMIZATION #
                # i.e. PCN predictive units are updated to minimize current free energy loss
                curr_fe.backward(retain_graph=False)
                inner_opt.step()
                inner_opt.zero_grad()
                self.zero_grad()
                # print(f"New prediction units {self.pred_units[1][0:9]}")
            else:

                curr_fe, in_preds = self.InferCycle(counter)
                last_in_preds = in_preds
                last_F.append(curr_fe)

            # print(f"Inferential cycle {counter + 1}; current free energy: {curr_fe}")
            # print(f'At iteration {counter}, total FE {curr_fe}')
            # print(f'At iteration {counter}, top neural activity {self.pred_units[-1]}')

            counter += 1

        ### Debugging ####
        # print(f'Number of inferntial cycles: {counter}')
        ### End of debug ###
        self.total_batch_fe += last_F[-1]

        # Logging some data from modules
        # NOTE: using -2 because the last module in self.pcn_layers is a dummy error module
        pred_units_zero = self.pcn_layers[0].units
        pred_units_top = self.pcn_layers[-2].units
        # print(f"Pred units top: {pred_units_top}")
        act_units_top = self.pcn_layers[-2].act_units

        return (
            last_F[-1],
            last_in_preds,
            pred_units_zero,
            pred_units_top,
            act_units_top,
            counter,
        )

    def InferCycle(self, counter):
        """Method that implements the inferential cycle during which the PCN minimizes free energy
        layer by layer.

        Arguments:

        - counter

        Output:

        - inf_fe (scalar): network energy free (total prediction error) for jsut completed inference cycle
        - in_preds (Pytorch tensor): predictions of the top layer

        """

        # TODO: deal with the supervised learning case, the top layer needs an input, so we need
        # to use the mode flag to request and pass the input when doing supervised learning

        in_preds = None
        inf_fe = torch.tensor([0.0]).to(DEV)

        for l in range(0, len(self.pcn_layers) - 2, 2):
            # Computing prediction from layer l+1 to layer l and corresponding prediction error.
            u_top = self.pcn_layers[l + 2](self.af_str)
            # Observations/data or predictive units activity is just copied
            i_bot = self.pcn_layers[l].get_units()
            # Saving predictions when they are of the input observations
            if l == 0:
                in_preds = u_top
            # print(f"i_bot size is {i_bot.size()}")
            # print(f"u_top bot size is {u_top.size()}")
            # Feeding input from below and predictions from above to prediction error module
            inf_fe += self.pcn_layers[l + 1](i_bot, u_top)

        return inf_fe, in_preds

    def learn(self, loss, num_batch, outer_opt):
        """
        Method to update PCN weights. We are changing the weights proportionally to the gradient of F
        in order to minimize it.

        Inputs:

        - num_batch (integer): batch number
        - outer_opt (optimizer): Pytorch optimizer object (e.g. SGD)
        """

        ##### CODE FOR MINI-BATCH GRADIENT DESCENT #####
        # print("Starting learning...")
        # self.total_batch_fe.backward(retain_graph=False)
        # # lf.backward(retain_graph=False)
        # outer_opt.step()
        # outer_opt.zero_grad()
        # self.zero_grad()

        ##### CODE FOR STOCHASTIC GRADIENT DESCENT #####
        # print("Starting learning...")
        loss = loss / self.bs
        loss.backward(retain_graph=False)
        # lf.backward(retain_graph=False)
        outer_opt.step()
        outer_opt.zero_grad()
        self.zero_grad()

    def reset(self, local):
        """
        Method to reset some network attributes after processing a single
        datapoint, or after a batch of datapoints.

        Input:

        - local (Boolean): variable to determine whether the reset is supposed
        to involve gradients or not.
        """

        pass

        if local:

            # Resetting after processing a single datapoint
            # self.errors = []
            # self.error_units = []
            # self.pred_units = nn.ParameterList([])
            # self.activ_units = []

            pass

        else:

            # Resetting after processing a full batch
            # self.errors = []
            # self.error_units = []
            # self.pred_units = nn.ParameterList([])
            # self.activ_units = []
            self.total_batch_fe = torch.zeros(1).to(DEV)


class PcnPreds(nn.Module):
    """Module of predictive units in a predictive coding network (PCN)"""

    def __init__(
        self,
        size,
        target_dim,
        batch_size,
        fix_activity,
        mode,
        input_layer=False,
    ):
        super().__init__()
        """
        Init method. Arguments:

        - size (integer): number of predictive nodes/units for a certain layer in the PCN
        - target_dim (integer): dimension of the target to be predicted;
        - batch_size (integer): number of datapoints in one (mini)-batch
        - fix_activity (Boolean): flag to indicate whether the units are subject to activity updates,
        - input_layer (Boolean): flag to indicate whether the units work as input layer (in such case there are no parameter and there is forward method)
        these units could represent either top level priors or bottom level inputs.
        """

        self.input_layer = input_layer
        self.batch_size = batch_size
        self.act_units = torch.empty((size, self.batch_size))
        self.fix_activity = (fix_activity, mode)

        self.activations = nn.ModuleDict(
            {
                "identity": nn.Identity(),
                "sigmoid": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "relu": nn.ReLU(),
                "lkrelu": nn.LeakyReLU(),
                "softplus": nn.Softplus(),
                "gelu": nn.GELU(),
            }
        )

        if self.fix_activity and not self.input_layer:

            self.units = torch.zeros((size, self.batch_size))
            self.weights = nn.Parameter(torch.empty(target_dim, size))
            self.bias = nn.Parameter(torch.empty((target_dim, self.batch_size)))

        elif self.input_layer and self.fix_activity:

            self.units = torch.empty((size, self.batch_size))

        elif self.input_layer and not self.fix_activity:

            self.units = nn.Parameter(torch.empty((size, self.batch_size)))

        else:

            self.units = nn.Parameter(torch.empty((size, self.batch_size)))
            self.weights = nn.Parameter(torch.empty(target_dim, size))
            self.bias = nn.Parameter(torch.empty((target_dim, self.batch_size)))

    @property
    def fix_activity(self):
        """ """
        return self._fix_activity

    @fix_activity.setter
    def fix_activity(self, val):
        """ """
        fix_act, mode = val
        print(f"Setting constraint on units update to {fix_act} in {mode} mode.")

        if fix_act == False and self.input_layer == True and mode == "supervised":

            self._fix_activity = fix_act
            print("Coverting input layer units into parameters...")
            # TODO: should make this depend on a size attribute instead of self.act_units
            self.units = nn.Parameter(torch.empty_like(self.act_units))

        else:

            self._fix_activity = fix_act

    def set_units(self, input):
        """Method to feed observations/datapoints to bottom units or to set priors for top layer."""

        if self.fix_activity == True:
            # Checking if self.units is a tensor of parameters
            # NOTE: e.g., deals with the case of a top prior used at test time after unsupervised training
            if self.units in self.parameters():
                # self.units was a Tensor of parameter so we have to assign an input as follows
                # NOTE: but it won't be updated because self.fix_activity is True
                with torch.no_grad():
                    self.units = nn.Parameter(input)
            else:
                # self.units was not a Tensor of parameters so the assignment works as follows
                # again, the tensor is not updated during inference
                self.units = input

        else:
            # Units are parameters and are initialized with an input
            # TODO: check if this might be redundant or not! (nope, used in supervised mode with top labels)
            with torch.no_grad():
                self.units = nn.Parameter(input)

    def get_units(self):
        """Method to return the current self.units parameters. This is used to provide the "input from below"
        to error modules, for the computation of prediction error."""

        return self.units

    def forward(self, act):
        """Computation of predictions happens here.
        Inputs:

        - act (string): string indicating the non-linear activation function to be used (e.g. relu);

        """
        assert self.input_layer == False, print(
            "Using forward method for input layer is not allowed."
        )

        if self.fix_activity:

            assert torch.sum(self.units) != 0.0, print(
                "This layer requires its activity being fixed by an input but its units are empty."
            )

        self.act_units = self.activations[act](self.units).to(DEV)
        u = torch.matmul(self.weights, self.act_units) + self.bias

        return u


class PcnErrs(nn.Module):
    """Module of error units in a predictive coding network (PCN)"""

    def __init__(
        self,
        size,
        batch_size,
        learn=False,
    ):
        super().__init__()
        """
        Init method. Arguments:

        - size (integer): number of error nodes/units for a certain layer in the PCN
        - batch_size (integer): number of datapoints in one (mini)-batch
        - learn (Boolean): flag for whether weights (covariance matrices) are learned or not
       """

        self.batch_size = batch_size
        self.units = torch.empty((size, self.batch_size))
        self.learn_weights = learn

        if self.learn_weights == False:
            self.weights = torch.eye(size)
        else:
            self.weights = nn.Parameter(torch.empty(size, size))

    def _invert_cov(self):
        """Compute precisions, i.e. inverse of covariance matrix or reciprocal of variance."""

        if self.weights.shape == (1, 1):
            # For 1-dim arrays simply compute the reciprocal (also, np.linalg.inv cannot deal with them).
            precisions = torch.tensor([1.0 / self.weights])

        else:
            # Computing matrix inverse with torch.linalg.inv
            precisions = torch.linalg.inv(self.weights)

        return precisions.to(DEV)

    def forward(self, bot_in, top_in, test=False):
        """Compute free energy for error units."""
        # print(bot_in.size())
        # print(top_in.size())
        self.units = bot_in - top_in
        precis = self._invert_cov()
        # print(self.units.size())
        # print(precis.size())
        f = torch.sum(torch.einsum("ik,ii,ik -> k", self.units, precis, self.units))
        # print(f"Free energies: {f.size()}")
        # print(f)

        return f

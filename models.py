import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils import MMDStatistic, MMDBiasedStatistic
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class MLPNet(nn.Module):
    """
    Vanilla multi-layer perceptron for regression.
    """
    def __init__(self, configs):
        super(MLPNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final regression layer.
        self.output_layer = nn.Linear(self.num_neurons[-1], 1)
        # whether use sigmoid in the last layer
        self.use_sigmoid = configs["use_sigmoid"]
        if self.use_sigmoid:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        out = self.output_layer(h_relu)
        if self.use_sigmoid:
            out = self.sigmoid_layer(out)
        return out

class WassersteinNet(nn.Module):
    """
    Multi-layer perceptron for regression with feature as one of the outputs when forwarding
    """
    def __init__(self, configs):
        super(WassersteinNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final regression layer.
        self.output_layer = nn.Linear(self.num_neurons[-1], 1)
        # Parameter of the adversary regression layer.
        self.num_adversaries = [self.num_neurons[-1]+1] + configs["adversary_layers"] # adv input dimension = hidden + ydim
        self.num_adversaries_layers = len(configs["adversary_layers"]) 
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                          for i in range(self.num_adversaries_layers)])
        self.sensitive_output_layer = nn.Linear(self.num_adversaries[-1], 1)
        # whether use sigmoid in the last layer
        self.use_sigmoid = configs["use_sigmoid"]
        if self.use_sigmoid:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, inputs, targets):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # regression
        out = self.output_layer(h_relu)
        if self.use_sigmoid:
            out = self.sigmoid_layer(out)
        # Adversary component.
        # include targets in the component
        h_relu = torch.cat((h_relu, targets), dim=-1)
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))
        adversary_out = self.sensitive_output_layer(h_relu)
        return out, adversary_out

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        out = self.output_layer(h_relu)
        if self.use_sigmoid:
            out = self.sigmoid_layer(out)
        return out

class CENet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for fairness.
    """
    def __init__(self, configs):
        super(CENet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final regression layer.
        self.output_layer = nn.Linear(self.num_neurons[-1], 1)
        # Parameter of the adversary regression layer.
        self.num_adversaries = [self.num_neurons[-1]+1] + configs["adversary_layers"] # adv input dimension = hidden + ydim
        self.num_adversaries_layers = len(configs["adversary_layers"])
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                          for i in range(self.num_adversaries_layers)])
        self.sensitive_output_layer = nn.Linear(self.num_adversaries[-1], 2)
        # whether use sigmoid in the last layer
        self.use_sigmoid = configs["use_sigmoid"]
        if self.use_sigmoid:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, inputs, targets):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # regression
        out = self.output_layer(h_relu)
        if self.use_sigmoid:
            out = self.sigmoid_layer(out)
        # Adversary classification component.
        # include targets in the component
        h_relu = torch.cat((h_relu, targets), dim=-1)
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))
        adversary_out = F.log_softmax(self.sensitive_output_layer(h_relu), dim=1)
        return out, adversary_out

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        out = self.output_layer(h_relu)
        if self.use_sigmoid:
            out = self.sigmoid_layer(out)
        return out

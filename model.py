"""Model using SGLD"""

from utils import SGLD, log_gaussian_loss, get_kl_divergence
import torch.nn as nn
import torch

class Langevin_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Langevin_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))

    def forward(self, x):
        return torch.mm(x, self.weights) + self.biases


class Langevin_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(Langevin_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = Langevin_Layer(input_dim, num_units)
        self.layer2 = Langevin_Layer(num_units, 2 * output_dim)

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)

        return x
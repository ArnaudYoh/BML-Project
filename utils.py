"""Utils for the model based on SGLD"""

from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import torch

class SGLD(Optimizer):
    """Optimizer based on Langevin Dynamics"""

    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.weight_decay = weight_decay

        args = {
            lr: lr,
            weight_decay: weight_decay,
        }

        super(SGLD, self).__init__(params, args)

    def step(self, closure=None):

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            l_r = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data

                if len(param.shape) == 1 and param.shape[0] == 1:
                    param.data.add_(-l_r, d_p)

                else:
                    if weight_decay != 0:
                        d_p.add(weight_decay, param.data)

                    unit_noise = Variable(param.data.new(param.size()).normal_())

                    param.data.add_(-l_r, 0.5 * d_p + unit_noise / l_r ** 0.5)


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)

    return - (log_coeff + exponent).sum()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik * (varpost_loglik - prior_loglik)).sum()

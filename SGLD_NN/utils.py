"""Utils for the model based on SGLD"""

from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np


class SGLD(Optimizer):
    """Optimizer based on Langevin Dynamics"""

    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.weight_decay = weight_decay

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
        )

        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
    """Computes the Gaussian loss between two sets of values of the same size"""
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)

    return - (log_coeff + exponent).sum()


def to_variable(var=(), cuda=True, volatile=False):
    """Transforms variables into Torch Variable Tensors"""
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        if cuda and not v.is_cuda:
            v = v.cuda()

        out.append(v)
    return out


def draw_train_data(x_train, y_train):
    """Draw the train_data and save the plot"""
    _, ax = plt.subplots()
    plt.style.use('default')
    plt.scatter(x_train, y_train, s=10, marker='x', color='black', alpha=0.5, label='samples')

    ax.legend()
    plt.xlim([-5, 5])
    plt.ylim([-5, 7])
    plt.xlabel('$x$', fontsize=30)
    plt.title('SGLD', fontsize=40)
    plt.tick_params(labelsize=30)
    plt.xticks(np.arange(-4, 5, 2))
    plt.yticks(np.arange(-4, 7, 2))
    plt.gca().set_yticklabels([])
    plt.gca().yaxis.grid(alpha=0.3)
    plt.gca().xaxis.grid(alpha=0.3)

    plt.savefig('./SGLD_NN/train.pdf', bbox_inches='tight')
    plt.close()


def draw_loss_over_iteration(loss_over_iter, batch_number, n_epochs, save_dir, record_period):
    """Draw the evolution of the loss over the iterations"""
    plt.figure(figsize=(8, 7))
    plt.style.use('default')
    for j in range(batch_number):
        plt.scatter(range(1, n_epochs + 1, record_period), loss_over_iter[j], s=5, marker='o', color='blue',
                    alpha=0.5, label='loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.title('Train Loss over # of Epochs', fontsize=30)
    plt.ylim((-400, 400))

    plt.savefig(save_dir + '/train_loss_iteration' + str(batch_number) + '.pdf')
    plt.close()


def draw_learned_dist(x_train, y_train, means, aleatoric, total_unc, dir_name, i):
    """Draws the learning distribution"""
    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    f, ax = plt.subplots()
    plt.style.use('default')
    plt.scatter(x_train, y_train, s=10, marker='x', color='black', alpha=0.5, label='samples')
    plt.fill_between(np.linspace(-5, 5, means.shape[0]), means + aleatoric, means + total_unc, color=c[0],
                     alpha=0.3,
                     label='Epistemic + Aleatoric')
    plt.fill_between(np.linspace(-5, 5, means.shape[0]), means - total_unc, means - aleatoric, color=c[0], alpha=0.3)
    plt.fill_between(np.linspace(-5, 5, means.shape[0]), means - aleatoric, means + aleatoric, color=c[1], alpha=0.4,
                     label='Aleatoric')
    plt.plot(np.linspace(-5, 5, means.shape[0]), means, color='black', linewidth=1,
             label='mean')
    ax.legend()
    plt.xlim([-5, 5])
    plt.ylim([-5, 7])
    plt.xlabel('$x$', fontsize=30)
    plt.title('SGLD', fontsize=40)
    plt.tick_params(labelsize=30)
    plt.xticks(np.arange(-4, 5, 2))
    plt.yticks(np.arange(-4, 7, 2))
    plt.gca().set_yticklabels([])
    plt.gca().yaxis.grid(alpha=0.3)
    plt.gca().xaxis.grid(alpha=0.3)

    plt.savefig(dir_name + '/sgld' + str(i) + '.pdf', bbox_inches='tight')
    plt.close()

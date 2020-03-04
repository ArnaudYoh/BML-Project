"""Model using SGLD"""

from utils import SGLD, log_gaussian_loss, to_variable
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import GPy

#torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device = 'cpu'

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
    def __init__(self, input_dim, output_dim, num_units, init_log_noise):
        super(Langevin_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = Langevin_Layer(input_dim, num_units)
        self.layer2 = Langevin_Layer(num_units, 2 * output_dim)

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)
        self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)

        return x


class Langevin_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, init_log_noise,
                 weight_decay):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.network = Langevin_Model(input_dim=input_dim,
                                      output_dim=output_dim,
                                      num_units=no_units,
                                      init_log_noise=init_log_noise,
                                      )
        if torch.device == "cuda":
            self.network.cuda()

        self.optimizer = SGLD(self.network.parameters(), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = log_gaussian_loss

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=torch.device == "cuda")

        # reset gradient and total loss
        self.optimizer.zero_grad()

        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)

        loss.backward()
        self.optimizer.step()

        return loss


if __name__ == "__main__":
    np.random.seed(2)
    no_points = 400
    lengthscale = 1
    variance = 1.0
    sig_noise = 0.3
    x = np.random.uniform(-3, 3, no_points)[:, None]
    x.sort(axis=0)

    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    C = k.K(x, x) + np.eye(no_points) * sig_noise ** 2

    y = np.random.multivariate_normal(np.zeros((no_points)), C)[:, None]
    y = (y - y.mean())
    x_train = x[75:325]
    y_train = y[75:325]

    best_net, best_loss = None, float('inf')
    num_nets, nets, losses = 50, [], []
    mix_epochs, burnin_epochs = 100, 3000
    num_epochs = mix_epochs * num_nets + burnin_epochs

    num_epochs, batch_size, nb_train = 8000, len(x_train), len(x_train)

    net = Langevin_Wrapper(input_dim=1, output_dim=1, no_units=200, learn_rate=1e-4,
                           batch_size=batch_size, no_batches=1, init_log_noise=0, weight_decay=1)

    for i in range(num_epochs):

        loss = net.fit(x_train, y_train)

        if i % 200 == 0:
            print('Epoch: %4d, Train loss = %8.3f' % (i, loss.cpu().data.numpy()))

        if i % mix_epochs == 0 and i > burnin_epochs: nets.append(copy.deepcopy(net.network))

    print("Using %d networks for prediction" % len(nets))
    samples = []
    noises = []
    for network in nets:
        preds = network.forward(torch.linspace(-5, 5, 200).cuda()).cpu().data.numpy()
        samples.append(preds)
        noises.append(torch.exp(network.log_noise).cpu().data.numpy())

    samples = np.array(samples)
    noises = np.array(noises).reshape(-1)
    means = (samples.mean(axis=0)).reshape(-1)

    aleatoric = noises.mean()
    epistemic = (samples.var(axis=0) ** 0.5).reshape(-1)
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(6, 5))
    plt.style.use('default')
    plt.scatter(x_train, y_train, s=10, marker='x', color='black', alpha=0.5)
    plt.fill_between(np.linspace(-5, 5, 200), means + aleatoric, means + total_unc, color=c[0], alpha=0.3,
                     label='Epistemic + Aleatoric')
    plt.fill_between(np.linspace(-5, 5, 200), means - total_unc, means - aleatoric, color=c[0], alpha=0.3)
    plt.fill_between(np.linspace(-5, 5, 200), means - aleatoric, means + aleatoric, color=c[1], alpha=0.4,
                     label='Aleatoric')
    plt.plot(np.linspace(-5, 5, 200), means, color='black', linewidth=1)
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
    plt.savefig('sgld.pdf', bbox_inches='tight')

    plt.show()

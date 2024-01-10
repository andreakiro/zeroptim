import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as D

from zeroptim.models.mlp import MLP
from zeroptim.optim.mezo import MeZO
from zeroptim.optim.smartes import SmartES

__supported_models__ = {
    'mlp': MLP,
}

__supported_activations__ = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
    'softmax': nn.Softmax,
}

__supported_optims__ = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'mezo': MeZO,
    'smartes': SmartES,
}

__supported_criterions__ = {
    'cross_entropy': nn.CrossEntropyLoss,
    'square_loss': nn.MSELoss,
}

__supported_datasets__ = {
    'cifar10': D.CIFAR10,
    'cifar100': D.CIFAR100,
    'mnist-digits': D.MNIST,
    'mnist-fashion': D.FashionMNIST,
    'svhn': D.SVHN
}
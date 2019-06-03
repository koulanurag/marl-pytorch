import os
import torch
from torch.utils.tensorboard import SummaryWriter


class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, path=None):
        """
        :param env: instance of the environment
        """
        self.env_fn = env_fn
        self.env = env_fn()
        self.env.seed(0)

        self.model = model_fn().to(device)
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Writer will output to ./runs/ directory by default
        path = os.path.join(path, 'runs')
        self.writer = SummaryWriter(path if path is not None else ())

    def act(self, state, debug=False):
        """ returns greedy action for the state"""
        raise NotImplementedError

    def train(self):
        """ trains the algorithm for given episodes"""
        raise NotImplementedError

    def save(self, path):
        """ save relevant properties in given path"""
        raise NotImplementedError

    def restore(self, path):
        """ save relevant properties from given path"""
        raise NotImplementedError

    def __del__(self):
        # self.writer.export_scalars_to_json()
        self.writer.close()

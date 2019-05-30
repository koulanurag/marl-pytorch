import torch


class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env, model, lr, discount, batch_size, device):
        """
        :param env: instance of the environment
        """

        self.env = env
        self.env.seed(0)

        self.model = model
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

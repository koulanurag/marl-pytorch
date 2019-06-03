import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, path=None):
        """
        :param env: instance of the environment
        """
        self.env_fn = env_fn
        self.env = env_fn()
        self.env.seed(0)
        self.episode_max_steps = 10000

        self.model = model_fn().to(device)
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # logging + visualization
        self.path = os.path.join(path, self.__class__.__name__, 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        self.model_path = os.path.join(self.path, 'model.p')
        self.writer = SummaryWriter(self.path)

    def act(self, state, debug=False):
        """ returns greedy action for the state"""
        raise NotImplementedError

    def save(self):
        """ save relevant properties in given path"""
        torch.save(self.model.state_dict(), self.model_path)

    def restore(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def close(self):
        self.writer.export_scalars_to_json(os.path.join(self.path, 'summary.json'))
        self.writer.close()
        print('saved')

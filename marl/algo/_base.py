import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path):
        """
        :param env: instance of the environment
        """
        self.env_fn = env_fn
        self.env = env_fn()
        self.env.seed(0)
        self.train_episodes = train_episodes
        self.episode_max_steps = episode_max_steps

        self.model = model_fn().to(device)
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # logging + visualization
        self.path = os.path.join(path, self.__class__.__name__, 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        # self.path = os.path.join(path, self.__class__.__name__, 'runs', 'run_1')
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

    def train(self, test_interval=10):
        print('Training......')
        for ep in range(0, self.train_episodes, test_interval):
            train_score, train_loss = self._train(test_interval)
            test_score = self.test(5)
            self.save()
            print(
                '# {}/{} Loss: {} Train Score: {} Test Score: {}'.format(ep,
                                                                         self.train_episodes,
                                                                         train_loss,
                                                                         np.mean(train_score),
                                                                         np.mean(test_score)))


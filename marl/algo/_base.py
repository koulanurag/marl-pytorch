import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path,
                 run_i):
        """
        :param env: instance of the environment
        """
        self.env_fn = env_fn
        self.env = env_fn()
        self.env.seed(run_i)
        self.train_episodes = train_episodes
        self.episode_max_steps = episode_max_steps

        self.model = model_fn().to(device)
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # logging + visualization
        self.path = os.path.join(path, self.__class__.__name__, 'runs', 'run_{}'.format(run_i))
        self.best_model_path = os.path.join(self.path, 'model.p')
        self.last_model_path = os.path.join(self.path, 'last_model.p')

    def act(self, state, debug=False):
        """ returns greedy action for the state"""
        raise NotImplementedError

    def save(self, path):
        """ save relevant properties in given path"""
        torch.save(self.model.state_dict(), path)

    def restore(self):
        self.model.load_state_dict(torch.load(self.best_model_path))

    def __writer_close(self):
        self.writer.export_scalars_to_json(os.path.join(self.path, 'summary.json'))
        self.writer.close()
        print('saved')

    def close(self):
        self.env.close()

    def train(self, test_interval=50):
        self.writer = SummaryWriter(self.path, flush_secs=10)

        print('Training......')
        test_scores = []
        best_score = None
        for ep in range(0, self.train_episodes, test_interval):
            train_score, train_loss = self._train(test_interval)
            test_score = self.test(5, log=True)
            test_scores.append(test_score)

            train_score = sum(train_score)
            test_score = sum(test_score)
            if best_score is None or best_score <= test_score:
                self.save(self.best_model_path)
                best_score = test_score
                print('Best Model Saved!')

            print(
                '# {}/{} Loss: {} Train Score: {} Test Score: {}'.format(ep + test_interval,
                                                                         self.train_episodes,
                                                                         train_loss,
                                                                         train_score,
                                                                         test_score))
        # keeping a copy of last trained model
        self.save(self.last_model_path)
        self.__writer_close()

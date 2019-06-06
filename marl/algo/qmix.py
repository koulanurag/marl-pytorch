import torch
import random
from ._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import numpy as np


class QMIX(_Base):
    """QMIX + Double DQN + Prioritized Replay + Soft Target Updates"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        pass

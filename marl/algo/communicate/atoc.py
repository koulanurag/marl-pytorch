"""
    Algorithm: Attentional Communication Model (ATOC)
    Paper: https://arxiv.org/abs/1810.02912 (NIPS 2018)
"""
import torch
import numpy as np
from .._base import _Base
from marl.utils import PrioritizedReplayMemory, ReplayMemory, Transition, soft_update, onehot_from_logits, \
    gumbel_softmax
from marl.utils import OUNoise, LinearDecay
from torch.nn import MSELoss


class ATOC(_Base):
    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, discrete_action_space, path):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        pass

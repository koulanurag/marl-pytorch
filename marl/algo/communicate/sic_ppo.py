"""
    Algorithm: Sharing is Caring with PPO( SICAC) - Just a random name
"""
import torch
import random
from .._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import numpy as np


class SICPPO(_Base):
    """ PPO  , Sharing Thoughts( state info + action intention) of each agent
        + Double DQN + Prioritized Replay + Soft Target Updates"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

"""Algorithm: Multi Agent Deep Deterministic Policy Gradient"""

from ._base import _Base


class MADDPG(_Base):
    def __init__(self, env, max_episode_steps):
        super().__init__(env, max_episode_steps)

    def _update(self):
        pass

    def train(self):
        pass

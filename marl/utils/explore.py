class _LinearDecay:
    """ Linearly Decays epsilon for exploration between a range of episodes"""

    def __init__(self, min_eps, max_eps, total_episodes):
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.total_episodes = total_episodes
        self.curr_episodes = 0
        # Todo: make 0.7 available as parameter
        self._threshold_episodes = 0.7 * total_episodes
        self.eps = max_eps

    def update(self):
        self.curr_episodes += 1
        eps = self.max_eps * (self._threshold_episodes - self.curr_episodes) / self._threshold_episodes
        self.eps = max(self.min_eps, eps)
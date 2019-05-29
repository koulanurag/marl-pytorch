class _Base:
    """ Base Class for  Multi Agent Algorithms"""

    def __init__(self, env, max_episode_steps):
        """
        :param env: instance of the environment
        """
        self.env = env
        self.env.seed(0)
        self.actions = env.action_space.n
        self.max_episode_steps = max_episode_steps

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


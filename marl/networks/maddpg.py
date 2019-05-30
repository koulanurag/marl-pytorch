import torch.nn as nn

class MADDPGNet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super().__init__()
        self.n_agents = len(obs_space_n)

        for i in range(self.n_agents):
            setattr(self, 'agent_{}_net'.format(i),
                    nn.Sequential(nn.Linear(obs_space_n, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU()))
            setattr(self, 'agent_{}_actor'.format(i),
                    nn.Linear(64, action_space_n))
            setattr(self, 'agent_{}_critic'.format(i),
                    nn.Sequential(nn.Linear(obs_space_n * self.n_agents, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(),
                                  nn.Linear(64, 1)))

    def act(self, id):
        """ returns only actor for an agent"""
        x = getattr(self, 'agent_{}_net'.format(id), )
        x = getattr(self, 'agent_{}_actor'.format(id), )

        return x

    def forward(self, *input):
        actor, critic = None, None
        for i in range(self.n_agents):
            x = getattr(self, 'agent_{}_net'.format(i), input)
            pi = getattr(self, 'agent_{}_actor'.format(i), input)
            v = getattr(self, 'agent_{}_critic'.format(i), input)

            actor.append(pi)
            critic.append(v)

        info = {}
        return actor, critic, info
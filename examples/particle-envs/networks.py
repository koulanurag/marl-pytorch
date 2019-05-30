import torch.nn as nn


class DDPGAgent(nn.Module):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(num_in_pol, 128),
                                   nn.LeakyReLU(),
                                   nn.Linear(128, 64),
                                   nn.LeakyReLU(),
                                   nn.Linear(64, num_out_pol))

        self.critic = nn.Sequential(nn.Linear(num_in_critic, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, 1))


class MADDPGNet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super().__init__()

        self.n_agents = len(obs_space_n)

        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            # Todo: Correct dim of input to the critic
            setattr(self, agent_i, DDPGAgent(obs_space_n[i], action_space_n[i], len(obs_space_n) * len(action_space_n)))

    def agent(self, i):
        getattr(self, 'agent_{}'.format(i))


# class MADDPGNet(nn.Module):
#     def __init__(self, obs_space_n, action_space_n):
#         super().__init__()
#         self.n_agents = len(obs_space_n)
#
#         for i in range(self.n_agents):
#             setattr(self, 'agent_{}_net'.format(i),
#                     nn.Sequential(nn.Linear(obs_space_n, 128),
#                                   nn.LeakyReLU(),
#                                   nn.Linear(128, 64),
#                                   nn.LeakyReLU()))
#             setattr(self, 'agent_{}_actor'.format(i),
#                     nn.Linear(64, action_space_n))
#             setattr(self, 'agent_{}_critic'.format(i),
#                     nn.Sequential(nn.Linear(obs_space_n * self.n_agents, 256),
#                                   nn.LeakyReLU(),
#                                   nn.Linear(256, 128),
#                                   nn.LeakyReLU(),
#                                   nn.Linear(128, 64),
#                                   nn.LeakyReLU(),
#                                   nn.Linear(64, 1)))
#
#     def act(self, id):
#         """ returns only actor for an agent"""
#         x = getattr(self, 'agent_{}_net'.format(id), )
#         x = getattr(self, 'agent_{}_actor'.format(id), )
#
#         return x
#
#     def forward(self, *input):
#         actor, critic = None, None
#         for i in range(self.n_agents):
#             x = getattr(self, 'agent_{}_net'.format(i), input)
#             pi = getattr(self, 'agent_{}_actor'.format(i), input)
#             v = getattr(self, 'agent_{}_critic'.format(i), input)
#
#             actor.append(pi)
#             critic.append(v)
#
#         info = {}
#         return actor, critic, info


class VDNet(nn.Module):
    def __init__(self, input, actions):
        super().__init__()
        self.n_agents = len(input)

        for i in range(self.n_agents):
            setattr(self, 'agent_{}_net'.format(i),
                    nn.Sequential(nn.Linear(input, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(),
                                  nn.Linear(64, actions)))

    def act(self, id):
        """ returns only actor for an agent"""
        x = getattr(self, 'agent_{}_net'.format(id), )
        return x

    def forward(self, *input):
        critic = None, None
        for i in range(self.n_agents):
            q = getattr(self, 'agent_{}_net'.format(i), input)
            critic.append(q)
        info = {}
        return critic, info


class IQNet(nn.Module):
    def __init__(self, input, actions):
        super().__init__()
        self.n_agents = len(input)

        for i in range(self.n_agents):
            setattr(self, 'agent_{}_net'.format(i),
                    nn.Sequential(nn.Linear(input, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(),
                                  nn.Linear(64, actions)))

    def act(self, id):
        """ returns only actor for an agent"""
        x = getattr(self, 'agent_{}_net'.format(id), )
        return x

    def forward(self, *input):
        critic = None, None
        for i in range(self.n_agents):
            q = getattr(self, 'agent_{}_net'.format(i), input)
            critic.append(q)
        info = {}
        return critic, info

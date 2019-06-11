import torch
import torch.nn as nn


class DDPGCritic(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super().__init__()
        self.obs_x = nn.Sequential(nn.Linear(obs_space_n, 1024),
                                   nn.ReLU())

        self._critic = nn.Sequential(nn.Linear(action_space_n + 1024, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 1))

        self._critic[-1].weight.data.fill_(0)
        self._critic[-1].bias.data.fill_(0)

    def forward(self, obs_n, action_n):
        x = self.obs_x(obs_n)
        x = self._critic(torch.cat((action_n, x), dim=1))
        return x


class DDPGAgent(nn.Module):
    def __init__(self, num_in_pol, num_out_pol, comb_obs_space, comb_action_space):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        super().__init__()
        self.action_space = num_out_pol
        self.actor = nn.Sequential(nn.Linear(num_in_pol, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, num_out_pol))

        self.critic = DDPGCritic(comb_obs_space, comb_action_space)

        self.actor[-1].bias.data.fill_(0)


class MADDPGNet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super().__init__()

        self.n_agents = len(obs_space_n)
        comb_obs_space = sum(len(o) for o in obs_space_n)
        comb_action_space = sum(a.n for a in action_space_n)
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, DDPGAgent(len(obs_space_n[i]), action_space_n[i].n,
                                             comb_obs_space, comb_action_space))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


class VDAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.action_space = action_space

        # self._critic = nn.Sequential(
        #     nn.Linear(obs_space, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space))

        self._critic = nn.Sequential(nn.Linear(obs_space, action_space))

        self._critic[-1].weight.data.fill_(0)
        self._critic[-1].bias.data.fill_(0)

    def forward(self, x):
        return self._critic(x)


class VDNet(nn.Module):
    def __init__(self, obs_space_n, action_space_n):
        super().__init__()

        self.n_agents = len(obs_space_n)
        for i in range(self.n_agents):
            agent_i = 'agent_{}'.format(i)
            setattr(self, agent_i, VDAgent(len(obs_space_n[i]), action_space_n[i].n))

    def agent(self, i):
        return getattr(self, 'agent_{}'.format(i))


class IDQNet(VDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

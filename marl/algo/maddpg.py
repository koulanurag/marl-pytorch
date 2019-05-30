"""Algorithm: Multi Agent Deep Deterministic Policy Gradient"""
import torch

from ._base import _Base
from marl.utils import ReplayMemory, Transition, soft_update
from torch.nn import MSELoss, SmoothL1Loss


class MADDPG(_Base):
    def __init__(self, env, model, lr, discount, batch_size, device, mem_len, tau):
        super().__init__(env, model, lr, discount, batch_size, device)
        self.memory = ReplayMemory(mem_len)
        self.tau = tau
        self.total_episodes = 10

        # Todo: Make copy instead of direct instance
        self.target_model = model
        self.target_model.eval()

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
        action_batch = torch.LongTensor(list(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(list(batch.reward)).to(self.device)
        next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
        non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)
        non_final_next_obs_batch = next_obs_batch[non_final_mask].to(self.device)

        # critic loss
        q_loss = 0
        for i in range(self.model.n_agents):
            pred_q_value = self.model.agent(i).critic()
            target_next_obs_q = torch.zeros(pred_q_value.shape).to(self.device)
            target_next_obs_q[non_final_mask] = self.target_model.agent(i).critic()
            target_q_value = (self.discount * target_next_obs_q) + reward_batch[i]
            q_loss += SmoothL1Loss(pred_q_value, target_q_value).mean()

        # actor loss
        # Todo: Complete this actor loss
        actor_loss = 0

        # Overall loss
        loss = actor_loss + q_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
        self.optimizer.step()

        # update target network
        soft_update(self.target_model, self.model, self.tau)

        return 0

    def __select_action(self, obs_n, explore=False):
        pass

    def train(self, episodes):
        self.model.train()
        for ep in range(episodes):
            done = False
            obs_n = self.env.reset()
            step = 0
            while not done:
                action_n = self._select_action(obs_n)
                next_obs_n, reward_n, done, info = self.env.step(action_n)

                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, done)

                obs_n = next_obs_n
                done = done or (step > self.max_episode_steps)
                step += 1

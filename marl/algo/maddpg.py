"""
    Algorithm: Multi Agent Deep Deterministic Policy Gradient
    Reference : https://github.com/shariqiqbal2810/maddpg-pytorch/
"""
import torch
import numpy as np
from ._base import _Base
from marl.utils import PrioritizedReplayMemory, ReplayMemory, Transition, soft_update, onehot_from_logits, \
    gumbel_softmax
from marl.utils import OUNoise, LinearDecay
from torch.nn import MSELoss


class MADDPG(_Base):
    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, discrete_action_space, path):
        """ Todo: Write note about usage or if anything specific is required to run"""
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        self.memory = ReplayMemory(mem_len)
        # self.memory = PrioritizedReplayMemory(mem_len)
        self.tau = tau

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.discrete_action_space = discrete_action_space
        if self.discrete_action_space:
            self.exploration = LinearDecay(0.1, 2, self.train_episodes)
        else:
            self.exploration = [OUNoise(self.model.agent(i).action_space) for i in range(self.model.n_agents)]

        self.__update_iter = 0

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.model.train()
        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        if self.batch_size > len(self.memory):
            self.model.eval()
            return None

        # transitions = self.memory.sample(self.batch_size)
        # Todo: move this beta in the Prioritized Replay memory
        # beta_start = 0.4
        # beta = min(1.0, beta_start + (self.__update_iter + 1) * (1.0 - beta_start) / 5000)

        # transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(list(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(list(batch.reward)).to(self.device)
        next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
        non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)
        # weights = torch.FloatTensor(weights).to(self.device)

        comb_obs_batch = obs_batch.flatten(1)
        comb_action_batch = action_batch.flatten(1)
        comb_next_obs_batch = next_obs_batch.flatten(1)

        # calculate loss
        q_loss_n, actor_loss_n = 0, 0
        # prios_n = 0
        for i in range(self.model.n_agents):
            # critic
            pred_q_value = self.model.agent(i).critic(comb_obs_batch, comb_action_batch)

            target_next_obs_q = torch.zeros(pred_q_value.shape).to(self.device)
            target_action_batch = self.__select_action(self.target_model, next_obs_batch)
            target_action_batch = target_action_batch.flatten(1).to(self.device)
            _next_q = self.target_model.agent(i).critic(comb_next_obs_batch, target_action_batch)
            target_next_obs_q[non_final_mask[:, i]] = _next_q[non_final_mask[:, i]]
            target_q_value = (self.discount * target_next_obs_q).squeeze(1) + reward_batch[:, i]
            q_loss = MSELoss()(pred_q_value.squeeze(1), target_q_value).mean()
            q_loss_n += q_loss

            # q_loss = (pred_q_value.squeeze(1) - target_q_value).pow(2) * weights
            # prios_n += q_loss + 1e-5
            # q_loss = q_loss.mean()
            # q_loss_n += q_loss

            # actor
            actor_i = self.model.agent(i).actor(obs_batch[:, i])
            _action_batch = action_batch.clone()
            if self.discrete_action_space:
                _action_batch[:, i] = gumbel_softmax(actor_i, hard=True)
            else:
                _action_batch[:, i] = actor_i

            _action_batch = _action_batch.flatten(1)
            actor_loss = - self.model.agent(i).critic(comb_obs_batch, _action_batch).mean()
            actor_loss += (actor_i ** 2).mean() * 1e-3
            actor_loss_n += actor_loss

            # log
            self.writer.add_scalar('agent_{}/critic_loss'.format(i), q_loss, self.__update_iter)
            self.writer.add_scalar('agent_{}/actor_loss'.format(i), actor_loss, self.__update_iter)

        # Overall loss
        loss = actor_loss_n + q_loss_n
        # prios_n /= self.model.n_agents

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        # self.memory.update_priorities(indices, prios_n.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        soft_update(self.target_model, self.model, self.tau)

        # log
        self.writer.add_scalar('_overall/critic_loss', q_loss_n, self.__update_iter)
        self.writer.add_scalar('_overall/actor_loss', actor_loss_n, self.__update_iter)

        # just keep track of update counts
        self.__update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return loss.item()

    def __select_action(self, model, obs_n, explore=False):
        act_n = []

        for i in range(model.n_agents):
            action = model.agent(i).actor(obs_n[:, i])
            if self.discrete_action_space:
                if explore:  # TODO: Exploration rate needs to be corrected over here
                    action = gumbel_softmax(action, temperature=self.exploration.eps, hard=True)
                else:
                    action = onehot_from_logits(action)
            else:  # continuous action
                if explore:
                    _noise = torch.FloatTensor(self.exploration[i].noise())
                    _noise.requires_grad = False
                    action += _noise

                action = action.clamp(-1, 1)

            act_n.append(action.unsqueeze(1))

        return torch.cat(act_n, dim=1)

    def _train(self, episodes):
        self.model.eval()
        train_rewards = []
        train_loss = []

        for ep in range(episodes):
            terminal = False
            obs_n = self.env.reset()
            step = 0
            ep_reward = [0 for _ in range(self.model.n_agents)]
            while not terminal:
                # self.env.render()

                torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                action_n = self.__select_action(self.model, torch_obs_n, explore=True)
                action_n = action_n.cpu().detach().numpy().tolist()[0]
                # print(action_n)
                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = all(done_n) or step >= self.episode_max_steps
                done_n = [terminal for _ in range(self.env.n_agents)]
                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, done_n)

                obs_n = next_obs_n
                step += 1
                if loss is not None:
                    train_loss.append(loss)

                for i, r_n in enumerate(reward_n):
                    ep_reward[i] += r_n

            train_rewards.append(ep_reward)
            if self.discrete_action_space:
                self.exploration.update()

            # log - training
            for i, r_n in enumerate(ep_reward):
                self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self.__update_iter)
            self.writer.add_scalar('_overall/train_reward', sum(ep_reward), self.__update_iter)
            self.writer.add_scalar('_overall/exploration_temperature', self.exploration.eps, self.__update_iter)

            print(ep, sum(ep_reward))

        return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])

    def test(self, episodes, render=False, log=False):
        self.model.eval()
        test_rewards = []
        with torch.no_grad():
            for ep in range(episodes):
                terminal = False
                obs_n = self.env.reset()
                step = 0

                ep_reward = [0 for _ in range(self.model.n_agents)]
                while not terminal:
                    if render:
                        self.env.render()

                    torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                    action_n = self.__select_action(self.model, torch_obs_n, explore=False)
                    action_n = action_n.cpu().numpy().tolist()[0]

                    next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                    terminal = all(done_n) or step >= self.episode_max_steps

                    obs_n = next_obs_n
                    step += 1
                    for i, r_n in enumerate(reward_n):
                        ep_reward[i] += r_n
                test_rewards.append(ep_reward)

            test_rewards = np.array(test_rewards).mean(axis=0)
            if log:
                # log - test
                for i, r_n in enumerate(test_rewards):
                    self.writer.add_scalar('agent_{}/eval_reward'.format(i), r_n, self.__update_iter)
                self.writer.add_scalar('_overall/eval_reward', sum(test_rewards), self.__update_iter)

                eval_obs = [[1, (2 + 1) / 8, 1, (5 + 1) / 8],
                            [1, (2 + 1) / 8, 0.5, (0 + 1) / 8],
                            [0.5, (2 + 1) / 8, 1, (5 + 1) / 8],
                            [1, (2 + 1) / 8, 0.5, (5 + 1) / 8]]

                for o_i, _obs in enumerate(eval_obs):
                    _obs = torch.Tensor(_obs).unsqueeze(0).to(self.device)
                    for agent_i in range(self.model.n_agents):
                        q = self.model.agent(agent_i).actor(_obs)
                        q = q.squeeze(0).cpu().numpy().tolist()
                        for i, x in enumerate(q):
                            self.writer.add_scalar('_agent_{}_pos_{}/action_{}'.format(agent_i, eval_obs[o_i], i),
                                                   x, self.__update_iter)

        return test_rewards

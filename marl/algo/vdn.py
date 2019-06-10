import torch
import random
from ._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import numpy as np


class VDN(_Base):
    """Value Decomposition Network + Double DQN + Prioritized Replay + Soft Target Updates"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        self.memory = PrioritizedReplayMemory(mem_len)
        self.tau = tau

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.exploration = LinearDecay(0.1, 1.0, self.train_episodes)
        self.__update_iter = 0

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.model.train()

        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        if self.batch_size > len(self.memory):
            self.model.eval()
            return None

        # Todo: move this beta in the Prioritized Replay memory
        beta_start = 0.4
        beta = min(1.0, beta_start + (self.__update_iter + 1) * (1.0 - beta_start) / 5000)

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(list(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(list(batch.reward)).to(self.device)
        next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)

        # calc loss
        overall_pred_q, target_q = 0, 0
        for i in range(self.model.n_agents):
            q_val_i = self.model.agent(i)(obs_batch[:, i])
            overall_pred_q += q_val_i.gather(1, action_batch[:, i, :].long())

            target_next_obs_q = torch.zeros(overall_pred_q.shape).to(self.device)
            non_final_next_obs_batch = next_obs_batch[:, i][non_final_mask[:, i]]

            # Double DQN update
            if not (non_final_next_obs_batch.shape[0] == 0):
                _max_actions = self.model.agent(i)(non_final_next_obs_batch).max(1, keepdim=True)[1].detach()
                _max_q = self.target_model.agent(i)(non_final_next_obs_batch).gather(1, _max_actions)
                target_next_obs_q[non_final_mask[:, i]] = _max_q

                target_q += target_next_obs_q.detach()

        target_q = (self.discount * target_q) + reward_batch.sum(dim=1, keepdim=True)
        loss = (overall_pred_q - target_q).pow(2) * weights.unsqueeze(1)
        prios = loss + 1e-5
        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        # Todo: Make 100 as a parameter
        # if self.__update_iter % 100:
        #     hard_update(self.target_model, self.model)
        soft_update(self.target_model, self.model, self.tau)

        # log
        self.writer.add_scalar('_overall/critic_loss', loss, self.__update_iter)
        self.writer.add_scalar('_overall/beta', beta, self.__update_iter)

        # just keep track of update counts
        self.__update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return loss.item()

    def __select_action(self, model, obs_n, explore=False):
        act_n = []

        for i in range(model.n_agents):
            """ selects epsilon greedy action for the state """
            one_hot_action = [0 for _ in range(model.agent(i).action_space)]
            if explore and self.exploration.eps > random.random():
                one_hot_action[random.randint(0, len(one_hot_action) - 1)] = 1
            else:
                one_hot_action[model.agent(i)(obs_n[:, i]).argmax(1).item()] = 1
            print(i, ' ', obs_n[:, i], model.agent(i)(obs_n[:, i]), one_hot_action)
            act_n.append(one_hot_action)

        # return torch.cat(act_n, dim=1)
        return torch.Tensor(act_n)

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
                action_n = action_n.cpu().detach().numpy().tolist()

                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = all(done_n) or step >= self.episode_max_steps
                action_n = [[_.index(1)] for _ in action_n]
                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, done_n)

                obs_n = next_obs_n
                step += 1
                if loss is not None:
                    train_loss.append(loss)

                for i, r_n in enumerate(reward_n):
                    ep_reward[i] += r_n

            train_rewards.append(ep_reward)
            self.exploration.update()

            # log - training
            for i, r_n in enumerate(ep_reward):
                self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self.__update_iter)
            self.writer.add_scalar('_overall/train_reward', sum(ep_reward), self.__update_iter)
            self.writer.add_scalar('_overall/exploration_rate', self.exploration.eps, self.__update_iter)

            # print(ep, sum(ep_reward))

        return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])

    def test(self, episodes, render=False, log=False):
        self.model.eval()
        with torch.no_grad():
            test_rewards = []
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
                    action_n = action_n.cpu().numpy().tolist()

                    # input_action = [int(x) for x in input('enter:')]
                    # action_n = np.zeros(np.array(action_n).shape).tolist()
                    # for i, a in enumerate(input_action):
                    #     action_n[i][a] = 1
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

        return test_rewards

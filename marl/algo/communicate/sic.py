"""
    Algorithm: Sharing is Caring ( SIC) - Just a random name
"""
import torch
import random
from .._base import _Base
from marl.utils import ReplayMemory, Transition, PrioritizedReplayMemory, soft_update, hard_update
from marl.utils import LinearDecay
from torch.nn import MSELoss
import numpy as np


class SIC(_Base):
    """ Value Based Methjod , Sharing Thoughts( state info + action intention) of each agent
        + Double DQN + Prioritized Replay + Soft Target Updates"""

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        """

        Args:
            env_fn: callback function returning instance of the environment
            model_fn: callback function returning instance of the model
            lr: learning rate
            discount: discount factor ( aka gamma)
            batch_size: Batch Size for training
            device: device to load data( gpu or cpu) during training
            mem_len: Size of Memory Buffer
            tau:
            train_episodes: No. of episodes for training
            episode_max_steps: Max. number of steps to be executed in the environment
            path: Path to store results
            log_suffix: Running index for logging
        """
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
        prios = 0
        overall_loss = 0

        obs_thoughts_batch, next_obs_thoughts_batch = None, None
        for i in range(self.model.n_agents):
            obs_thought = self.model.agent(i).get_message(obs_batch[:, i]).unsqueeze(1)
            next_obs_thought = self.model.agent(i).get_message(next_obs_batch[:, i]).unsqueeze(1)
            if i == 0:
                obs_thoughts_batch, next_obs_thoughts_batch = obs_thought, next_obs_thought
            else:
                obs_thoughts_batch = torch.cat((obs_thoughts_batch, obs_thought), dim=1)
                next_obs_thoughts_batch = torch.cat((next_obs_thoughts_batch, next_obs_thought), dim=1)

        for i in range(self.model.n_agents):

            if 0 < i < (self.model.n_agents - 1):
                obs_global_thoughts = torch.cat(obs_thoughts_batch[:, :i, :], obs_thoughts_batch[:, i + 1:, :])
                next_obs_global_thoughts = torch.cat(next_obs_thoughts_batch[:, :i, :],
                                                     next_obs_thoughts_batch[:, i + 1:, :])
            elif i == 0:
                obs_global_thoughts = obs_thoughts_batch[:, i + 1:, :]
                next_obs_global_thoughts = next_obs_thoughts_batch[:, i + 1:, :]
            else:
                obs_global_thoughts = obs_thoughts_batch[:, :i, :]
                next_obs_global_thoughts = next_obs_thoughts_batch[:, :i, :]

            q_val_i = self.model.agent(i)(obs_thoughts_batch[:, i, :], obs_global_thoughts)
            pred_q = q_val_i.gather(1, action_batch[:, i].unsqueeze(1).long())

            target_next_obs_q = torch.zeros(pred_q.shape).to(self.device)
            non_final_next_obs_thoughts = next_obs_thoughts_batch[:, i][non_final_mask]
            non_final_global_thoughts = next_obs_global_thoughts[non_final_mask]

            # Double DQN update
            target_q = 0
            if not (non_final_next_obs_thoughts.shape[0] == 0):
                _max_actions = self.model.agent(i)(non_final_next_obs_thoughts, non_final_global_thoughts)
                _max_actions = _max_actions.max(1, keepdim=True)[1].detach()

                _max_q = self.target_model.agent(i)(non_final_next_obs_thoughts, non_final_global_thoughts)
                _max_q = _max_q.gather(1, _max_actions)

                target_next_obs_q[non_final_mask] = _max_q

                target_q = target_next_obs_q.detach()

            target_q = (self.discount * target_q) + reward_batch.sum(dim=1, keepdim=True)
            loss = (pred_q - target_q).pow(2) * weights.unsqueeze(1)
            prios += loss + 1e-5
            loss = loss.mean()
            overall_loss += loss
            self.writer.add_scalar('agent_{}/critic_loss'.format(i), loss.item(), self.__update_iter)

        # Optimize the model
        self.optimizer.zero_grad()
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # update target network
        soft_update(self.target_model, self.model, self.tau)

        # log
        self.writer.add_scalar('_overall/critic_loss', overall_loss, self.__update_iter)
        self.writer.add_scalar('_overall/beta', beta, self.__update_iter)

        # just keep track of update counts
        self.__update_iter += 1

        # resuming the model in eval mode
        self.model.eval()

        return overall_loss.item()

    def __select_action(self, model, obs_n, explore=False):
        """ selects epsilon greedy action for the state """
        if explore and self.exploration.eps > random.random():
            act_n = self.env.action_space.sample()
        else:
            act_n, _thoughts = [], []
            for i in range(model.n_agents):
                _th = model.agent(i).get_message(obs_n[:, i])
                _thoughts.append(_th)

            _thoughts = torch.cat(_thoughts)
            for i in range(model.n_agents):
                _neighbours = list(range(model.n_agents))
                _neighbours.remove(i)
                _q_vals = model.agent(i)(_thoughts[i].unsqueeze(0), _thoughts[_neighbours, :].unsqueeze(0))
                act_n.append(_q_vals.argmax(1).item())

        return act_n

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
                torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                action_n = self.__select_action(self.model, torch_obs_n, explore=True)

                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = all(done_n) or step >= self.episode_max_steps
                done_n = [terminal for _ in range(self.env.n_agents)]
                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, terminal)

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

            print(ep, sum(ep_reward))

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

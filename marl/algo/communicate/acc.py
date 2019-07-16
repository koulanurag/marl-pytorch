"""
    Algorithm: Actor Critic With Communication of hidden state

    On-policy method
"""
import torch
from .._base import _Base
import numpy as np
import torch.nn.functional as F
from ma_gym.wrappers import Monitor
import os

class ACC(_Base):

    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, train_episodes,
                 episode_max_steps, path):
        """
        Actor Critic With Communication of hidden state

        Args:
            env_fn: callback function returning instance of the environment
            model_fn: callback function returning instance of the model
            lr: learning rate
            discount: discount factor ( aka gamma)
            batch_size: No. of rollouts to perform before update of the network
            device: device to load data( gpu or cpu) during training
            mem_len: Size of Memory Buffer
            tau:
            train_episodes: No. of episodes for training
            episode_max_steps: Max. number of steps to be executed in the environment
            path: Path to store results
            log_suffix: Running index for logging
        """
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, train_episodes, episode_max_steps, path)
        self.tau = tau
        self.__update_iter = 0
        self.__episode_iter = 0
        self.entropy_coef = 0.01
        self.critic_loss_coef = 0.5
        self.truncate_n = 1
        self.gae_lambda = 1

        self.n_trajectory_info = []

    def _update(self):
        critic_loss = [0 for _ in range(self.model.n_agents)]
        policy_loss = [0 for _ in range(self.model.n_agents)]

        for trajectory_info in self.n_trajectory_info:
            obs, _rewards, _critic, _log_probs, _entropies = trajectory_info

            R = torch.zeros(1, 1).to(self.device)
            gae = [torch.zeros(1, 1).to(self.device) for _ in range(self.model.n_agents)]
            _critic.append([torch.zeros(1, 1).to(self.device) for _ in range(self.model.n_agents)])
            for step in reversed(range(len(_rewards))):
                step_reward = sum(_rewards[step])  # each agent maximizes team reward rather than local reward
                R = self.discount * R + step_reward
                for agent_i in range(self.model.n_agents):
                    advantage = R - _critic[step][agent_i]
                    critic_loss[agent_i] += 0.5 * advantage.pow(2)

                    # Generalized Advantage Estimation
                    delta_t = step_reward
                    delta_t += self.discount * _critic[step + 1][agent_i].data - _critic[step][agent_i].data
                    gae[agent_i] = gae[agent_i] * self.discount * self.gae_lambda + delta_t

                    policy_loss[agent_i] -= _log_probs[step][agent_i] * gae[agent_i].detach() - \
                                            self.entropy_coef * _entropies[step][agent_i]

        critic_loss = [c / self.batch_size for c in critic_loss]
        policy_loss = [p / self.batch_size for p in policy_loss]

        self.optimizer.zero_grad()
        loss = (sum(policy_loss) + self.critic_loss_coef * sum(critic_loss))
        loss.backward()
        self.optimizer.step()

        # log
        for agent_i in range(self.model.n_agents):
            self.writer.add_scalar('agent_{}/policy_loss'.format(agent_i), policy_loss[agent_i].item(),
                                   self._step_iter)
            self.writer.add_scalar('agent_{}/critic_loss'.format(agent_i), critic_loss[agent_i].item(),
                                   self._step_iter)
        self.writer.add_scalar('_overall/critic_loss', sum(critic_loss).item(),
                               self._step_iter)
        self.writer.add_scalar('_overall/actor_loss', sum(policy_loss).item(),
                               self._step_iter)
        self.writer.add_scalar('_overall/loss', loss.item(),
                               self._step_iter)

        return loss.item()

    def _train(self, episodes):

        train_rewards = []
        train_loss = []

        for ep in range(episodes):
            ep_log_probs, ep_rewards, ep_critic_info, ep_entropies, ep_obs = [], [], [], [], []

            terminal = False
            obs_n = self.env.reset()
            step = 0
            log_ep_reward = [0 for _ in range(self.model.n_agents)]

            self.model.init_hidden(device=self.device)
            while not terminal:
                ep_obs.append(obs_n)
                torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)

                thoughts = []
                for agent_i in range(self.model.n_agents):
                    thoughts.append(self.model.agent(agent_i).get_thought(torch_obs_n[:, agent_i]))
                thoughts = torch.stack(thoughts)

                action_n = []
                log_probs, critic_info, entropies, = [], [], []
                for agent_i in range(self.model.n_agents):
                    # assuming every other agent is a neighbour as of now
                    _neighbours = list(range(self.model.n_agents))
                    _neighbours.remove(agent_i)

                    logits, critic = self.model.agent(agent_i)(thoughts[_neighbours])
                    prob = F.softmax(logits, dim=1)
                    log_prob = F.log_softmax(logits, dim=1)
                    entropy = -(log_prob * prob).sum(1)

                    action = prob.multinomial(num_samples=1).detach()
                    log_prob = log_prob.gather(1, action)
                    action_n.append(action.item())

                    log_probs.append(log_prob)
                    critic_info.append(critic)
                    entropies.append(entropy)

                next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                terminal = all(done_n) or step >= self.episode_max_steps

                obs_n = next_obs_n
                step += 1
                self._step_iter += 1
                for i, r_n in enumerate(reward_n):
                    log_ep_reward[i] += r_n

                ep_log_probs.append(log_probs)
                ep_critic_info.append(critic_info)
                ep_entropies.append(entropies)
                ep_rewards.append(reward_n)

                if (self.truncate_n is not None) and (step % self.truncate_n == 0):
                    self.model.hidden_detach()

            train_rewards.append(log_ep_reward)
            self.__episode_iter += 1

            self.n_trajectory_info.append((ep_obs, ep_rewards, ep_critic_info, ep_log_probs, ep_entropies))
            if self.__episode_iter % self.batch_size == 0:
                train_loss.append(self._update())
                self.n_trajectory_info = []  # empty the trajectory info

            # log - training
            for i, r_n in enumerate(log_ep_reward):
                self.writer.add_scalar('agent_{}/train_reward'.format(i), r_n, self._step_iter)
            self.writer.add_scalar('_overall/train_reward', sum(log_ep_reward), self._step_iter)
            self.writer.add_scalar('_overall/train_ep_steps', step, self._step_iter)

            print(ep, sum(log_ep_reward))

        return np.array(train_rewards).mean(axis=0), (np.mean(train_loss) if len(train_loss) > 0 else [])

    def test(self, episodes, render=False, log=False, record=False):
        self.model.eval()

        env = self.env
        if record:
            env = Monitor(self.env_fn(), directory=os.path.join(self.path, 'recordings'), force=True,
                          video_callable=lambda episode_id: True)

        with torch.no_grad():
            test_rewards = []
            total_test_steps = 0
            for ep in range(episodes):
                terminal = False
                obs_n = self.env.reset()
                step = 0
                ep_reward = [0 for _ in range(self.model.n_agents)]

                self.model.init_hidden(device=self.device)
                while not terminal:
                    if render:
                        self.env.render()

                    torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)

                    thoughts = []
                    for agent_i in range(self.model.n_agents):
                        thoughts.append(self.model.agent(agent_i).get_thought(torch_obs_n[:, agent_i]))
                    thoughts = torch.stack(thoughts)

                    action_n = []
                    for agent_i in range(self.model.n_agents):
                        # assuming every other agent is a neighbour as of now
                        _neighbours = list(range(self.model.n_agents))
                        _neighbours.remove(agent_i)

                        logits, critic = self.model.agent(agent_i)(thoughts[_neighbours])
                        prob = F.softmax(logits, dim=1)
                        action = prob.argmax(1).item()
                        # action = prob.multinomial(num_samples=1).detach()

                        if log and step == 0 and ep == 0:
                            log_prob = F.log_softmax(logits, dim=1)
                            entropy = -(log_prob * prob).sum(1)
                            self.writer.add_scalar('agent_{}/entropy'.format(agent_i), entropy, self._step_iter)

                        action_n.append(action)

                    next_obs_n, reward_n, done_n, info = self.env.step(action_n)
                    terminal = all(done_n) or step >= self.episode_max_steps

                    obs_n = next_obs_n
                    step += 1
                    for i, r_n in enumerate(reward_n):
                        ep_reward[i] += r_n

                total_test_steps += step
                test_rewards.append(ep_reward)

            test_rewards = np.array(test_rewards).mean(axis=0)

            # log - test
            if log:
                for i, r_n in enumerate(test_rewards):
                    self.writer.add_scalar('agent_{}/eval_reward'.format(i), r_n, self._step_iter)
                self.writer.add_scalar('_overall/eval_reward', sum(test_rewards), self._step_iter)
                self.writer.add_scalar('_overall/test_ep_steps', total_test_steps / episodes, self._step_iter)

        if record:
            env.close()
        return test_rewards

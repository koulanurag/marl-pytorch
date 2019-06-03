"""
    Algorithm: Multi Agent Deep Deterministic Policy Gradient
    Reference : https://github.com/shariqiqbal2810/maddpg-pytorch/
"""
import torch

from ._base import _Base
from marl.utils import ReplayMemory, Transition, soft_update, onehot_from_logits, gumbel_softmax
from torch.nn import MSELoss, SmoothL1Loss


class MADDPG(_Base):
    def __init__(self, env_fn, model_fn, lr, discount, batch_size, device, mem_len, tau, path=None):
        super().__init__(env_fn, model_fn, lr, discount, batch_size, device, path)
        self.memory = ReplayMemory(mem_len)
        self.tau = tau
        self.total_episodes = 10

        self.target_model = model_fn().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.__update_iter = 0

    def __update(self, obs_n, action_n, next_obs_n, reward_n, done):
        self.memory.push(obs_n, action_n, next_obs_n, reward_n, done)

        loss = torch.Tensor([0])
        if self.batch_size < len(self.memory):
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            obs_batch = torch.FloatTensor(list(batch.state)).to(self.device)
            action_batch = torch.FloatTensor(list(batch.action)).to(self.device)
            reward_batch = torch.FloatTensor(list(batch.reward)).to(self.device)
            next_obs_batch = torch.FloatTensor(list(batch.next_state)).to(self.device)
            non_final_mask = 1 - torch.ByteTensor(list(batch.done)).to(self.device)

            comb_obs_batch = obs_batch.flatten(1)
            comb_action_batch = action_batch.flatten(1)
            comb_next_obs_batch = next_obs_batch.flatten(1)

            # calc loss
            q_loss_n, actor_loss_n = 0, 0
            for i in range(self.model.n_agents):
                # critic
                pred_q_value = self.model.agent(i).critic(comb_obs_batch, comb_action_batch)
                # Todo: Improve over here for processing only non-terminal states
                target_next_obs_q = torch.zeros(pred_q_value.shape).to(self.device)
                target_action_batch = self.__select_action(self.target_model, next_obs_batch)
                target_action_batch = target_action_batch.flatten(1).to(self.device)
                target_next_obs_q[non_final_mask] = self.target_model.agent(i).critic(comb_next_obs_batch,
                                                                                      target_action_batch)
                target_q_value = (self.discount * target_next_obs_q).squeeze(1) + reward_batch[:, i]
                q_loss = SmoothL1Loss()(pred_q_value.squeeze(1), target_q_value).mean()
                q_loss_n += q_loss

                # actor
                actor_i = self.model.agent(i).actor(obs_batch[:, i])
                _action_batch = action_batch.clone()
                _action_batch[:, i] = gumbel_softmax(actor_i, hard=True)
                _action_batch = _action_batch.flatten(1)
                actor_loss = - self.model.agent(i).critic(comb_obs_batch, _action_batch).mean()
                actor_loss_n += actor_loss

                self.writer.add_scalars('agent_{}/losses'.format(i),
                                        {'critic_loss': q_loss_n, 'actor_loss': actor_loss_n},
                                        self.__update_iter)

            # Overall loss
            loss = actor_loss_n + q_loss_n

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()

            # update target network
            soft_update(self.target_model, self.model, self.tau)

            self.writer.add_scalars('overall/losses',
                                    {'critic_loss': q_loss, 'actor_loss': actor_loss},
                                    self.__update_iter)

            # just keep track of update counts
            self.__update_iter += 1

        return loss.item()

    def __select_action(self, model, obs_n, explore=False):
        act_n = []

        for i in range(model.n_agents):
            action = model.agent(i).actor(obs_n[:, i])
            action = onehot_from_logits(action, eps=(0.3 if explore else 0)).cpu()
            act_n.append(action.unsqueeze(1))

        return torch.cat(act_n, dim=1)

    def train(self, episodes):
        self.model.train()
        for ep in range(episodes):
            done = False
            obs_n = self.env.reset()
            step = 0
            while not done:
                torch_obs_n = torch.FloatTensor(obs_n).to(self.device).unsqueeze(0)
                action_n = self.__select_action(self.model, torch_obs_n, explore=True)
                action_n = action_n.cpu().numpy().tolist()[0]

                next_obs_n, reward_n, done, info = self.env.step(action_n)
                done = True in done

                loss = self.__update(obs_n, action_n, next_obs_n, reward_n, done)

                obs_n = next_obs_n
                step += 1

                print(loss)

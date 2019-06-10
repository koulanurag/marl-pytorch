from __future__ import absolute_import
import os
import argparse
import torch
import numpy as np
import gym

import marl
from marl.algo import MADDPG, VDN, IDQN

import ma_gym
from networks import MADDPGNet, VDNet, IDQNet

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Multi Agent Reinforcement Learning')
    parser.add_argument('--env', default='CrossOver-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Enforces no cuda usage (default: %(default)s)')
    parser.add_argument('--algo', choices=['maddpg', 'vdn', 'idqn'],
                        help='Training Algorithm', required=True)
    parser.add_argument('--train', action='store_true', default=False,
                        help='Trains the model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Evaluates the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--discount', type=float, default=0.95,
                        help=' Discount rate (or Gamma) for TD error (default: %(default)s)')
    parser.add_argument('--train_episodes', type=int, default=2000,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')

    args = parser.parse_args()
    device = 'cuda' if ((not args.no_cuda) and torch.cuda.is_available()) else 'cpu'
    args.env_result_dir = os.path.join(args.result_dir, args.env)
    if not os.path.exists(args.env_result_dir):
        os.makedirs(args.env_result_dir)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize environment
    env_fn = lambda: gym.make(args.env)
    env = env_fn()
    obs_n = env.reset()
    action_space_n = env.action_space

    # initialize algorithms
    if args.algo == 'maddpg':
        maddpg_net = lambda: MADDPGNet(obs_n, action_space_n)
        algo = MADDPG(env_fn, maddpg_net, lr=args.lr, discount=args.discount, batch_size=args.batch_size,
                      device=device, mem_len=50000, tau=0.01, path=args.env_result_dir, discrete_action_space=True,
                      train_episodes=args.train_episodes, episode_max_steps=5000)
    elif args.algo == 'vdn':
        vdnet_fn = lambda: VDNet(obs_n, action_space_n)
        algo = VDN(env_fn, vdnet_fn, lr=args.lr, discount=args.discount, batch_size=args.batch_size,
                   device=device, mem_len=10000, tau=0.01, path=args.env_result_dir,
                   train_episodes=args.train_episodes, episode_max_steps=5000)
    elif args.algo == 'idqn':
        iqnet_fn = lambda: IDQNet(obs_n, action_space_n)
        algo = IDQN(env_fn, iqnet_fn, lr=args.lr, discount=args.discount, batch_size=args.batch_size,
                   device=device, mem_len=10000, tau=0.01, path=args.env_result_dir,
                   train_episodes=args.train_episodes, episode_max_steps=5000)

    # The real game begins!! Broom, Broom, Broommmm!!
    try:
        if args.train:
            algo.train()
        if args.test:
            algo.restore()
            test_score = algo.test(episodes=10, render=True, log=False)
            print(test_score)
    finally:
        algo.close()
    env.close()

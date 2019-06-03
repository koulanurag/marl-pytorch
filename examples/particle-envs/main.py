from __future__ import absolute_import
import os
import argparse
import torch
import numpy as np
import marl
from marl.algo import MADDPG, VDN, IQL

from make_env import make_env
from networks import MADDPGNet, VDNet, IQNet

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Multi Agent Reinforcement Learning')
    parser.add_argument('--env', default='simple_spread',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Enforces no cuda usage (default: %(default)s)')
    parser.add_argument('--algo', choices=['maddpg', 'vdn', 'iql'],
                        help='Training Algorithm', required=True)
    parser.add_argument('--train', action='store_true', default=False,
                        help='Evaluates the discrete model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Evaluates the discrete model')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--discount', type=float, default=0.95,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP address for hosting the data (default: %(default)s)')
    parser.add_argument('--port', default='8052',
                        help='hosting port (default:%(default)s)')
    parser.add_argument('--visualize_results', action='store_true', default=False,
                        help='Visualizes the results in the browser (default: %(default)s)')

    args = parser.parse_args()
    device = 'cuda' if ((not args.no_cuda) and torch.cuda.is_available()) else 'cpu'
    args.env_result_dir = os.path.join(args.result_dir, args.env)
    if not os.path.exists(args.env_result_dir):
        os.makedirs(args.env_result_dir)

    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize environment
    env_fn = lambda: make_env(args.env)
    env = env_fn()
    obs_n = env.reset()
    action_space_n = env.action_space

    # initialize algorithms
    if args.algo == 'maddpg':
        maddpg_net = lambda: MADDPGNet(obs_n, action_space_n)
        algo = MADDPG(env_fn, maddpg_net, lr=args.lr, discount=args.discount, batch_size=args.batch_size,
                      device=device, mem_len=10000, tau=0.01, path=args.env_result_dir)
    elif args.algo == 'vdn':
        vdnet = lambda: VDNet()
        algo = VDN(env_fn, vdnet)
    elif args.algo == 'iql':
        iqnet = lambda: IQNet()
        algo = IQL(env_fn, iqnet)

    if args.train:
        algo.train(episodes=10)
    if args.test:
        marl.test()

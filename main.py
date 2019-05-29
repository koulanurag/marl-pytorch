import os
import argparse
import torch
from marl.algo import iql

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Multi Agent Reinforcement Learning')
    parser.add_argument('--env', default='CartPole-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Enforces no cuda usage (default: %(default)s)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Evaluates the discrete model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Evaluates the discrete model')
    parser.add_argument('--lr', type=float, default=0.001,
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

    # Process arguments and create relative paths to store results
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    algo = iql()

    if args.train:
        pass
    if args.test:
        pass
    if args.visualize_results:
        pass

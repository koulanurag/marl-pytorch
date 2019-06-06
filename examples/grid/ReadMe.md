# grid world envs 

These environments are copied from [here.](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment)

#### Environment Installation
```python install -r env_requirements.txt```

### Usage
Please refer to following:
```bash
usage: main.py [-h] [--env ENV] [--result_dir RESULT_DIR] [--no_cuda] --algo
               {maddpg,vdn,iql} [--train] [--test] [--lr LR]
               [--discount DISCOUNT] [--train_episodes TRAIN_EPISODES]
               [--batch_size BATCH_SIZE] [--seed SEED]

Multi Agent Reinforcement Learning

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Name of the environment (default: CartPole-v0)
  --result_dir RESULT_DIR
                        Directory Path to store results (default: <current working directory>)
  --no_cuda             Enforces no cuda usage (default: False)
  --algo {maddpg,vdn,iql}
                        Training Algorithm
  --train               Evaluates the discrete model
  --test                Evaluates the discrete model
  --lr LR               Learning rate (default: 0.001)
  --discount DISCOUNT   Learning rate (default: 0.95)
  --train_episodes TRAIN_EPISODES
                        Learning rate (default: 2000)
  --batch_size BATCH_SIZE
                        Learning rate (default: 128)
  --seed SEED           seed (default: 0)
```
Example: ``` python main.py --env CartPole-v0 --algo vdn --train```

Example: ``` python main.py --env CartPole-v0 --algo maddpg --train```
    
### Visualize Training
```bash
tensorboard --logdir=results/<env_name>/<algo_name>/runs
```

Example: ```tensorboard --logdir=results/CartPole-v0/VDN/runs```
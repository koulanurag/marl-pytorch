# Grid environment

It's used for evaluation of the marl algorithms over single agent environment. This is done for debugging purposes

### Usage
Argument Usage could be found by :```python main.py --help```

Example: ``` python main.py --env CrossOver-v0 --algo vdn --train```

Example: ``` python main.py --env CrossOver-v0 --algo maddpg --train```
    
### Visualize Training
```bash
tensorboard --logdir=results/<env_name>/<algo_name>/runs
```

Example: ```tensorboard --logdir=results/CrossOver-v0/VDN/runs```
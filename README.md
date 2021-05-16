# marl-pytorch

Implementations of multi agent reinforcement learning algorithms in pytorch

**[Status: Archived | No Longer Maintained | Code provided as it is]**

### Algorithms :
* VDN : Value Decomposition Network
* MADDPG : Multi Agent Deep Deterministic Policy Gradient
* IDQN : Independent Q Learning


## Installation

```bash
pip install -r requirements.txt
python setup.py install # use 'develop' instead of 'install' if developing the package
```

## Usage

```python
>>> from marl.algo import VDN
>>> env_fn = lambda : ... # returns instance of the env
>>> model_fn = lambda : ... # return instance of the model
>>> algo = VDN(env_fn,model_fn)
>>> algo.train()
```
Please refer to examples for detailed usage

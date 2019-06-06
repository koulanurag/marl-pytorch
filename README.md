# marl-pytorch

Implementations of multi agent reinforcement learning algorithms in pytorch

### Algorithms :
* VDN : Value Decomposition Network
* MADDPG : Multi Agent Deep Deterministic Policy Gradient
* IQL : (Pending)
## Installation

```bash
pip install -r requirements.txt
python setup.py install # use 'develop' instead of 'install' if developing the package
```

## Usage

```python
>>> from marl.algo import VDN
>>> algo = VDN(env_fn,model_fn)
>>> algo.train()
```
Please refer to examples for detailed usage

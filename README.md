# marl-pytorch

Implementations of multi agent reinforcement learning algorithms in pytorch

### Algorithms :
* VDN : Value Decomposition Network
* IQL : Independent Q Learning
* MADDPG : Multi Agent Deep Deterministic Policy Gradient

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage

```python
>>> from marl.algo import VDN
>>> algo = VDN(env_fn,model)
>>> algo.train()
```
Please refer to examples for detailed usage

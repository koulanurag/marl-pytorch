import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CrossOver-v0',
    entry_point='ma_gym.envs.crossover:CrossOver',
)

register(
    id='CrossOver-v1',
    entry_point='ma_gym.envs.crossover:CrossOverF',
)
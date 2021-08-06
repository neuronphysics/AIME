import gym
from envs import register_envs

register_envs()

def load_data(env_name, mode=None):
    raise NotImplementedError

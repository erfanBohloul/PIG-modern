import numpy as np
import gymnasium as gym
import os, sys
from arguments_ddpg import get_args
from algos.ddpg_agent import ddpg_agent
from goal_env import *
from goal_env.mujoco import *
import random
import torch
from gymnasium import Wrapper


def get_env_params(env):
    """Get environment parameters using modern Gymnasium API"""
    obs, info = env.reset()  # Modern reset returns (obs, info)
    params = {
        "obs": obs["observation"].shape[0],
        "goal": obs["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": float(env.action_space.high[0]),  # Convert to float
    }
    params["max_timesteps"] = env.spec.max_episode_steps if hasattr(env, 'spec') else 200
    return params


def launch(args):
    # Create the environments
    env = gym.make(args.env_name)
    test_env = gym.make(args.test_env)  # Changed from args.test to args.test_env
    
    # Set random seeds for reproduce
    # Modern Gymnasium uses seed parameter in reset, not env.seed()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":  # Fixed comparison
        torch.cuda.manual_seed(args.seed)
    
    # Get the environment parameters
    env_params = get_env_params(env)
    env_params["max_test_timesteps"] = test_env.spec.max_episode_steps if hasattr(test_env, 'spec') else 500
    
    # Create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params, test_env)
    ddpg_trainer.learn()


if __name__ == "__main__":
    # Get the params
    args = get_args()
    launch(args)
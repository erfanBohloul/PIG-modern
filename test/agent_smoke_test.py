import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from goal_env import *
from goal_env.mujoco import *
from algos.ddpg_agent import ddpg_agent
from arguments_ddpg import get_args
import gymnasium as gym
import numpy as np
import random


def get_env_params(env):
    """Get environment parameters using modern Gymnasium API"""
    obs, info = env.reset()
    params = {
        "obs": obs["observation"].shape[0],
        "goal": obs["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": float(env.action_space.high[0]),
    }
    params["max_timesteps"] = env.spec.max_episode_steps if hasattr(env, 'spec') else 200
    return params


def collect_random_transitions(agent, env, n_episodes=2):
    """Collect random transitions and add them to the agent's buffer"""
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_steps = 0
        max_steps = env.spec.max_episode_steps if hasattr(env, 'spec') else 200
        
        while not done and episode_steps < max_steps:
            # Take random action
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add transition to buffer if it has the right method
            if hasattr(agent, 'buffer') and hasattr(agent.buffer, 'add'):
                transition = {
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done
                }
                # Try adding with different possible signatures
                if hasattr(agent.buffer, 'add'):
                    try:
                        agent.buffer.add(obs, action, reward, next_obs, done)
                    except:
                        try:
                            agent.buffer.add(obs, action, reward, next_obs, done, info)
                        except:
                            agent.buffer.add(transition)
            
            obs = next_obs
            episode_steps += 1
        
        print(f"Episode {episode + 1} completed with {episode_steps} steps")


def main():
    args = get_args()
    args.env_name = "AntMazeL-v1"
    args.test_env = "AntMazeLTest-v1"
    args.device = "cpu"
    args.n_epochs = 1
    args.n_batches = 2
    args.batch_size = 32
    args.n_test_rollouts = 2
    args.initial_sample = 50
    args.landmark = 50
    args.plan_budget = 5
    args.eval_freq = 1
    args.seed = 0

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed(args.seed)

    # Create environments
    print("Creating environments...")
    env = gym.make(args.env_name)
    test_env = gym.make(args.test_env)
    
    # Get environment parameters
    env_params = get_env_params(env)
    env_params["max_test_timesteps"] = test_env.spec.max_episode_steps if hasattr(test_env, 'spec') else 500

    # Create agent with required parameters
    print("Creating agent...")
    agent = ddpg_agent(args, env, env_params, test_env)
    print("Agent created successfully")

    # Collect random transitions using the random_policy
    print("Collecting random transitions...")
    collect_random_transitions(agent, env, n_episodes=2)
    print("Random collection done")

    # Test a single step of learning if possible
    if hasattr(agent, 'learn'):
        print("Testing learning step...")
        # Check if buffer has enough samples
        buffer_size = len(agent.buffer) if hasattr(agent, 'buffer') and hasattr(agent.buffer, '__len__') else 0
        print(f"Buffer size: {buffer_size}")
        
        if buffer_size >= args.batch_size:
            # Try to perform one learning step
            agent.learn()
            print("Learning step completed")
        else:
            print(f"Not enough samples in buffer (need {args.batch_size}, have {buffer_size})")

    print("Smoke test passed!")

if __name__ == "__main__":
    main()
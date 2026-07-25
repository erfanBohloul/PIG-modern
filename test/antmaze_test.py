import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import numpy as np
from goal_env import *
from goal_env.mujoco import *

env = gym.make("AntMazeL-v1")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, info = env.reset()
print("\nReset observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))
print("Observation shapes:")
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"  {k}: {np.array(v).shape}")
else:
    print("  ", np.array(obs).shape)

action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print("\nStep results:")
print("  reward:", reward)
print("  terminated:", terminated)
print("  truncated:", truncated)
print("  info keys:", info.keys() if isinstance(info, dict) else info)

env.close()
print("\n✅ Basic environment test passed!")
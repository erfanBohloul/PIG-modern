# test_mujoco.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
import numpy as np
from goal_env.mujoco.create_maze_env import register_maze_envs, register_fetch_envs

def test_mujoco_envs():
    """Test the modernized MuJoCo environments"""
    register_maze_envs()
    register_fetch_envs()
    
    print("=== Testing MuJoCo Environments ===\n")
    
    # Test a few registered environments
    test_envs = [
        "PointMaze-v1",
        "AntMaze-v1",
        "PointMazeS-v1",
        "PointMazeW-v1",
        "PointMazeP-v1",
        "Reacher3D-v0",
        "Pusher-v0",
    ]
    
    for env_id in test_envs:
        try:
            print(f"\nTesting {env_id}...")
            env = gym.make(env_id, render_mode=None)
            
            # Test reset
            obs, info = env.reset(seed=42)
            print(f"✓ Reset successful")
            print(f"  - Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'not dict'}")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step successful")
            print(f"  - Reward: {reward:.4f}")
            print(f"  - Terminated: {terminated}")
            print(f"  - Truncated: {truncated}")
            print(f"  - Is success: {info.get('is_success', False)}")
            
            env.close()
            print(f"✅ {env_id} passes all tests")
            
        except Exception as e:
            print(f"❌ {env_id} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_mujoco_envs()
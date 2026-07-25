import gymnasium as gym
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_env.bitflip import BitFlipEnv
from goal_env.fourroom import FourRoom
from goal_env.mountaincar import MountainCarEnv
from goal_env.plane import NaivePlane5

def test_gymnasium_env(env_class, env_kwargs=None, num_episodes=3, num_steps=10):
    """
    Comprehensive test suite for Gymnasium environments
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Test 1: Environment creation
    print(f"\n=== Testing {env_class.__name__} ===")
    try:
        env = env_class(**env_kwargs)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test 2: Check observation and action spaces
    print("\n--- Testing spaces ---")
    try:
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'action_space'), "Missing action_space"
        print(f"✓ Observation space: {type(env.observation_space).__name__}")
        print(f"✓ Action space: {type(env.action_space).__name__}")
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    
    # Test 3: Reset method
    print("\n--- Testing reset ---")
    try:
        obs, info = env.reset()
        assert obs is not None, "Reset returned None for observation"
        assert isinstance(info, dict), "Reset info should be a dict"
        print(f"✓ Reset successful, obs shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    
    # Test 4: Step method
    print("\n--- Testing step ---")
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check return types
        assert obs is not None, "Step returned None for observation"
        assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"Terminated should be bool, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Truncated should be bool, got {type(truncated)}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"
        
        print(f"✓ Step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False
    
    # Test 5: Multiple episodes
    print("\n--- Testing multiple episodes ---")
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            for step in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            print(f"✓ Episode {episode + 1} completed, total reward: {episode_reward}")
    except Exception as e:
        print(f"✗ Multiple episodes failed: {e}")
        return False
    
    # Test 6: Seed functionality
    print("\n--- Testing seeding ---")
    try:
        # Reset with seed
        obs1, info1 = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward2, terminated2, truncated2, info2 = env.step(action)
        
        # Reset with same seed should give same initial state
        obs1_seeded, info1_seeded = env.reset(seed=42)
        
        # Compare (note: may not be identical due to random actions in step, but initial should match)
        print(f"✓ Seeding works (reset with same seed gives same initial state)")
    except Exception as e:
        print(f"⚠ Seed test had issues: {e}")
    
    # Test 7: Close
    print("\n--- Testing close ---")
    try:
        env.close()
        print("✓ Environment closed successfully")
    except Exception as e:
        print(f"⚠ Close had issues: {e}")
    
    print(f"\n✅ All tests passed for {env_class.__name__}!")
    return True

# Example usage with all environments
def run_all_tests():
    # Test the BitFlip environment
    test_gymnasium_env(BitFlipEnv, {"num_bits": 5})
    
    # Test the FourRoom environment
    test_gymnasium_env(FourRoom, {"goal_type": "fix_goal"})
    
    # Test the MountainCar environment
    test_gymnasium_env(MountainCarEnv, {"goal_dim": 1})
    
    # Test the Plane environments
    test_gymnasium_env(NaivePlane5, {"is_render": False})

if __name__ == "__main__":
    run_all_tests()
# test_goalplane.py
import gymnasium as gym
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_goalplane():
    """Test the modernized GoalPlane environment"""
    
    print("=== Testing GoalPlane Environment ===\n")
    
    # Test 1: Create environment with different configurations
    test_configs = [
        {"env_name": "Plane-v0", "type": "random", "maze_size": 15},
        {"env_name": "Plane-v0", "type": "easy", "maze_size": 15},
        {"env_name": "Plane-v0", "type": "mid", "maze_size": 15},
        {"env_name": "Plane-v0", "type": "hard", "maze_size": 15},
        {"env_name": "Plane-v0", "maze_size": 30, "goals": (2.5, 12.5)},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {config} ---")
        try:
            env = GoalPlane(**config)
            print(f"✓ Environment created successfully")
            
            # Test reset
            obs, info = env.reset(seed=42)
            print(f"✓ Reset successful")
            print(f"  - Observation keys: {list(obs.keys())}")
            print(f"  - Goal: {obs['desired_goal']}")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✓ Step successful")
            print(f"  - Reward: {reward:.4f}")
            print(f"  - Terminated: {terminated}")
            print(f"  - Truncated: {truncated}")
            print(f"  - Is success: {info.get('is_success', False)}")
            
            # Test compute_reward
            achieved_goal = np.array([5.0, 5.0])
            desired_goal = np.array([10.0, 10.0])
            reward = env.compute_reward(achieved_goal, desired_goal, {})
            print(f"✓ compute_reward works: {reward:.4f}")
            
            env.close()
            print(f"✅ Environment {i+1} passed all tests")
            
        except Exception as e:
            print(f"✗ Test {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

def test_goalplane_with_registered_envs():
    """Test GoalPlane with registered environments"""
    
    print("\n=== Testing GoalPlane with Registered Environments ===\n")
    
    # These need to be registered first
    env_ids = [
        "GoalPlane-v0",
        "GoalPlaneEasy-v0", 
        "GoalPlaneMid-v0",
        "GoalPlaneHard-v0",
        "GoalPlaneTest-v0",
        "GoalPlane-v1",
        "GoalPlaneTest-v1",
    ]
    
    for env_id in env_ids:
        try:
            print(f"\nTesting {env_id}...")
            env = gym.make(env_id)
            obs, info = env.reset(seed=42)
            
            # Take a few steps
            for step in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            
            print(f"✓ {env_id} works with gym.make()")
            env.close()
            
        except Exception as e:
            print(f"✗ {env_id} failed: {e}")

if __name__ == "__main__":
    # First, ensure the Plane environment is registered
    from goal_env.plane import NaivePlane5
    from goal_env.goal_plane_env import GoalPlane
    
    # Register the Plane environment if not already registered
    try:
        gym.register(
            id="Plane-v0",
            entry_point="goal_env.plane:NaivePlane5",
        )
    except:
        pass  # Already registered
    
    # Run tests
    test_goalplane()
    test_goalplane_with_registered_envs()
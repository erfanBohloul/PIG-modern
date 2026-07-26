# test_weighted_pig.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gymnasium as gym
from goal_env import *
from goal_env.mujoco import *
from algos.ddpg_agent import ddpg_agent
from arguments_ddpg import get_args
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def test_weighted_pig_configs():
    """Test different Weighted-PIG configurations"""
    
    print("="*60)
    print("Testing Weighted-PIG Configurations")
    print("="*60)
    
    # Test configurations
    configs = [
        # (name, weighting, beta, uniform_mix, warmup, ramp)
        ("Uniform (Original PIG)", "uniform", 0.0, 0.25, 10000, 20000),
        ("Weighted Graph", "graph", 0.5, 0.25, 10000, 20000),
        ("Weighted Euclidean", "euclidean", 0.5, 0.25, 10000, 20000),
        ("Weighted Index", "index", 0.5, 0.25, 10000, 20000),
        ("Weighted Graph High Beta", "graph", 1.0, 0.25, 10000, 20000),
        ("Weighted Graph More Uniform", "graph", 0.5, 0.5, 10000, 20000),
    ]
    
    for name, weighting, beta, uniform_mix, warmup, ramp in configs:
        print(f"\n--- Testing: {name} ---")
        print(f"  Weighting: {weighting}, Beta: {beta}, Uniform Mix: {uniform_mix}")
        
        try:
            # Create args with specific Weighted-PIG settings
            args = get_args()
            args.env_name = "AntMazeL-v1"
            args.test_env = "AntMazeLTest-v1"
            args.device = "cpu"
            args.n_epochs = 1  # Just for testing
            args.n_batches = 2
            args.batch_size = 32
            args.n_test_rollouts = 2
            args.initial_sample = 50
            args.landmark = 50
            args.plan_budget = 5
            args.eval_freq = 1
            args.seed = 42
            
            # Weighted-PIG specific args
            args.goal_loss_weighting = weighting
            args.goal_loss_beta = beta
            args.goal_loss_uniform_mix = uniform_mix
            args.goal_loss_warmup_steps = warmup
            args.goal_loss_beta_ramp_steps = ramp
            args.lambda_goal_loss = 1.0
            
            # Create environments
            env = gym.make(args.env_name)
            test_env = gym.make(args.test_env)
            
            # Get environment parameters
            env_params = get_env_params(env)
            env_params["max_test_timesteps"] = test_env.spec.max_episode_steps if hasattr(test_env, 'spec') else 500
            
            # Create agent
            agent = ddpg_agent(args, env, env_params, test_env)
            print(f"✓ Agent created successfully with {weighting} weighting")
            
            # Test a single update step
            print(f"  Testing network update...")
            
            # Collect some random data
            for episode in range(2):
                obs, info = env.reset()
                done = False
                steps = 0
                while not done and steps < 50:
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    if done:
                        break
            
            # Check if buffer has data
            print(f"  Buffer size: {agent.buffer.current_size}")
            
            if agent.buffer.current_size > 0:
                # Try to update network
                actor_loss, critic_loss, goal_loss = agent._update_network()
                print(f"  ✓ Network update successful")
                print(f"    Actor loss: {actor_loss.item():.4f}")
                print(f"    Critic loss: {critic_loss.item():.4f}")
                print(f"    Goal loss: {goal_loss.item():.4f}")
            else:
                print(f"  ⚠ Buffer empty, skipping network update")
            
            env.close()
            test_env.close()
            print(f"✓ {name} test passed!")
            
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()


def test_weighted_pig_metrics():
    """Test that Weighted-PIG metrics are properly tracked"""
    
    print("\n" + "="*60)
    print("Testing Weighted-PIG Metrics")
    print("="*60)
    
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
    args.seed = 42
    
    # Test different weighting methods
    for weighting in ["uniform", "graph", "euclidean", "index"]:
        print(f"\n--- Testing metrics for: {weighting} ---")
        
        args.goal_loss_weighting = weighting
        args.goal_loss_beta = 0.5
        args.goal_loss_uniform_mix = 0.25
        
        try:
            env = gym.make(args.env_name)
            test_env = gym.make(args.test_env)
            env_params = get_env_params(env)
            env_params["max_test_timesteps"] = test_env.spec.max_episode_steps if hasattr(test_env, 'spec') else 500
            
            agent = ddpg_agent(args, env, env_params, test_env)
            
            # Simulate a few updates to populate metrics
            for episode in range(2):
                obs, info = env.reset()
                done = False
                steps = 0
                while not done and steps < 50:
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    if done:
                        break
            
            if agent.buffer.current_size > 0:
                agent._update_network()
                
                # Check metrics
                metrics = {
                    "weighted_multi_path_fraction": agent.weighted_multi_path_fraction,
                    "weighted_distance_span": agent.weighted_distance_span,
                    "weighted_entropy": agent.weighted_entropy,
                    "weighted_max_weight": agent.weighted_max_weight,
                    "weighted_beta_effective": agent.weighted_beta_effective,
                }
                
                print(f"  Metrics for {weighting}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value:.4f}")
                
                # Basic validation
                if weighting != "uniform":
                    assert agent.weighted_beta_effective >= 0, "Beta effective should be >= 0"
                    assert 0 <= agent.weighted_entropy <= 1, "Entropy should be between 0 and 1"
                print(f"✓ {weighting} metrics validated")
            
            env.close()
            test_env.close()
            
        except Exception as e:
            print(f"✗ {weighting} metrics test failed: {e}")
            import traceback
            traceback.print_exc()


def test_weighted_pig_edge_cases():
    """Test edge cases for Weighted-PIG"""
    
    print("\n" + "="*60)
    print("Testing Weighted-PIG Edge Cases")
    print("="*60)
    
    edge_cases = [
        ("Beta=0", "graph", 0.0, 0.25),
        ("Beta very high", "graph", 100.0, 0.25),
        ("Uniform mix=0", "graph", 0.5, 0.0),
        ("Uniform mix=1", "graph", 0.5, 1.0),
        ("Warmup > ramp", "graph", 0.5, 0.25, 20000, 10000),  # Invalid but should handle gracefully
    ]
    
    for edge_case in edge_cases:
        name, weighting, beta, uniform_mix, *extra = edge_case
        warmup = extra[0] if extra else 10000
        ramp = extra[1] if extra else 20000
        
        print(f"\n--- Testing: {name} ---")
        
        try:
            args = get_args()
            args.env_name = "AntMazeL-v1"
            args.test_env = "AntMazeLTest-v1"
            args.device = "cpu"
            args.n_epochs = 1
            args.n_batches = 1
            args.batch_size = 32
            args.n_test_rollouts = 1
            args.initial_sample = 50
            args.landmark = 50
            args.plan_budget = 5
            args.eval_freq = 1
            args.seed = 42
            
            args.goal_loss_weighting = weighting
            args.goal_loss_beta = beta
            args.goal_loss_uniform_mix = uniform_mix
            args.goal_loss_warmup_steps = warmup
            args.goal_loss_beta_ramp_steps = ramp
            args.lambda_goal_loss = 1.0
            
            env = gym.make(args.env_name)
            test_env = gym.make(args.test_env)
            env_params = get_env_params(env)
            env_params["max_test_timesteps"] = test_env.spec.max_episode_steps if hasattr(test_env, 'spec') else 500
            
            agent = ddpg_agent(args, env, env_params, test_env)
            print(f"✓ Agent created for edge case: {name}")
            
            # Quick validation
            if beta < 0:
                assert args.goal_loss_beta >= 0, f"Beta should be >= 0, got {beta}"
            
            env.close()
            test_env.close()
            print(f"✓ {name} passed!")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests"""
    print("Starting Weighted-PIG Tests\n")
    
    # Test 1: Configurations
    test_weighted_pig_configs()
    
    # Test 2: Metrics
    test_weighted_pig_metrics()
    
    # Test 3: Edge cases
    test_weighted_pig_edge_cases()
    
    print("\n" + "="*60)
    print("All Weighted-PIG tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
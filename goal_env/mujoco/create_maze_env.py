from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from collections import OrderedDict
import gymnasium as gym
import numpy as np
import copy
from gymnasium import Wrapper
from gymnasium.envs.registration import EnvSpec


class GoalWrapper(Wrapper):
    """Modernized Goal Wrapper for MuJoCo Maze Environments"""
    
    def __init__(
        self, 
        env, 
        maze_size_scaling, 
        random_start, 
        low, 
        high, 
        maze_low, 
        maze_high,
        render_mode=None,
    ):
        super(GoalWrapper, self).__init__(env)
        
        # Store render mode
        self._render_mode = render_mode
        
        # Get observation space
        ob_space = env.observation_space
        self.maze_size_scaling = maze_size_scaling
        
        # Convert to numpy arrays with proper dtype
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        maze_low = np.array(maze_low, dtype=np.float32)
        maze_high = np.array(maze_high, dtype=np.float32)
        
        self.goal_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.maze_space = gym.spaces.Box(low=maze_low, high=maze_high, dtype=np.float32)

        self.goal_dim = low.size
        self.distance_threshold = 5 * maze_size_scaling / 8.0

        # Modern Gymnasium uses spaces.Dict
        self.observation_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "observation": ob_space,
                    "desired_goal": self.goal_space,
                    "achieved_goal": self.goal_space,
                }
            )
        )
        self.goal = None
        self.random_start = random_start

    def step(self, action):
        """Modern Gymnasium step method"""
        # Step the environment - returns 5 values in modern Gymnasium
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Get achieved goal from observation
        achieved_goal = observation[..., :self.goal_dim].astype(np.float32)
        
        # Create observation dict
        out = {
            "observation": observation.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32) if self.goal is not None else None,
            "achieved_goal": achieved_goal,
        }
        
        # Calculate reward
        reward = self.compute_rew(achieved_goal, self.goal, info)
        info["is_success"] = reward > -self.distance_threshold
        
        # Convert reward to Python float
        reward = float(reward)
        
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return out, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Modern Gymnasium reset method"""
        # Handle seeding
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self.env, 'reset'):
                self.env.reset(seed=seed)
        
        # Reset the environment
        observation = self.env.reset()
        
        # Sample a goal
        self.goal = self.goal_space.sample()
        while self.env._is_in_collision(self.goal):
            self.goal = self.goal_space.sample()
        
        # Random start position without collision
        if self.random_start:
            xy = self.maze_space.sample()
            while self.env._is_in_collision(xy):
                xy = self.maze_space.sample()
            self.env.wrapped_env.set_xy(xy)
            observation = self.env._get_obs()

        # Create observation dict
        achieved_goal = observation[..., :self.goal_dim].astype(np.float32)
        out = {
            "observation": observation.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
            "achieved_goal": achieved_goal,
        }
        
        # Modern Gymnasium returns (obs, info)
        return out, {}

    def compute_rew(self, state, goal, info):
        """Compute reward between state and goal"""
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)


def create_maze_env(
    env_name=None,
    top_down_view=False,
    maze_size_scaling=8,
    random_start=True,
    goal_args=[],
    maze_args=[],
    render_mode=None,
):
    """Factory function to create maze environment"""
    n_bins = 0
    manual_collision = False
    
    # Parse environment name
    if env_name.startswith("Ego"):
        n_bins = 8
        env_name = env_name[3:]
    
    if env_name.startswith("Ant"):
        manual_collision = True
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = maze_size_scaling
    elif env_name.startswith("Point"):
        cls = PointMazeEnv
        manual_collision = True
        env_name = env_name[5:]
        maze_size_scaling = maze_size_scaling
    else:
        assert False, "unknown env %s" % env_name

    # Set maze configuration
    maze_id = None
    observe_blocks = False
    put_spin_near_agent = False
    
    if env_name == "Maze":
        maze_id = "Maze"
    elif env_name == "Maze1":
        maze_id = "Maze1"
    elif env_name == "Push":
        maze_id = "Push"
    elif env_name == "Fall":
        maze_id = "Fall"
    elif env_name == "Block":
        maze_id = "Block"
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == "BlockMaze":
        maze_id = "BlockMaze"
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == "MazeL":
        maze_id = "MazeL"
    elif env_name == "MazeS":
        maze_id = "MazeS"
    elif env_name == "MazeW":
        maze_id = "MazeW"
    elif env_name == "MazeP":
        maze_id = "MazeP"
    else:
        raise ValueError("Unknown maze environment %s" % env_name)

    # Create MuJoCo environment kwargs
    gym_mujoco_kwargs = {
        "maze_id": maze_id,
        "n_bins": n_bins,
        "observe_blocks": observe_blocks,
        "put_spin_near_agent": put_spin_near_agent,
        "top_down_view": top_down_view,
        "manual_collision": manual_collision,
        "maze_size_scaling": maze_size_scaling,
    }
    
    # Create the environment
    gym_env = cls(**gym_mujoco_kwargs)
    gym_env.reset()
    
    # Scale goal and maze arguments
    goal_args = np.array(goal_args, dtype=np.float32) / 8 * maze_size_scaling
    maze_args = np.array(maze_args, dtype=np.float32) / 8 * maze_size_scaling

    # Wrap the environment
    return GoalWrapper(
        gym_env, 
        maze_size_scaling, 
        random_start, 
        *goal_args, 
        *maze_args,
        render_mode=render_mode,
    )

def register_maze_envs():
    """Register all maze environments"""
    robots = ["Point", "Ant"]
    task_types = [
        "Maze",
        "Maze1",
        "Push",
        "Fall",
        "Block",
        "BlockMaze",
        "MazeL",
        "MazeS",
        "MazeW",
        "MazeP",
    ]
    all_name = [x + y for x in robots for y in task_types]
    for name_t in all_name:
        for Test in ["", "Test"]:
            maze_args = [[-4, -4], [20, 20]]
            max_timestep = 200
            random_start = True
            if name_t[-4:] == "Maze" or name_t[-4:] == "aze1" or name_t[-4:] == "azeL":
                goal_args = [[-4, -4], [20, 20]]
            if Test == "Test":
                goal_args = [[0.0, 16.0], [1e-3, 16 + 1e-3]]
                if name_t[-4:] == "azeL":
                    goal_args = [[16.0, 16.0], [16 + 1e-3, 16 + 1e-3]]
                random_start = False
                max_timestep = 500

            if name_t[-4:] == "azeS":
                max_timestep = 400
                goal_args = [[-4, -4], [36, 36]]
                maze_args = [[-4, -4], [36, 36]]
                if Test == "Test":
                    goal_args = [[32.0, 32.0], [32 + 1e-3, 32 + 1e-3]]
                    random_start = False
                    max_timestep = 1000

            if name_t[-4:] == "azeW":
                max_timestep = 400
                goal_args = [[-4, -12], [36, 28]]
                maze_args = [[-4, -12], [36, 28]]
                if Test == "Test":
                    goal_args = [[0.0, 16.0], [0 + 1e-3, 16 + 1e-3]]
                    random_start = False
                    max_timestep = 1000

            if name_t[-4:] == "azeP":
                max_timestep = 400
                goal_args = [[-12, -4], [28, 36]]
                maze_args = [[-12, -4], [28, 36]]
                if Test == "Test":
                    goal_args = [[16.0, 0.0], [16.0 + 1e-3, 0.0 + 1e-3]]
                    random_start = False
                    max_timestep = 1000

            gym.register(
                id=name_t + Test + "-v0",
                entry_point="goal_env.mujoco.create_maze_env:create_maze_env",
                kwargs={
                    "env_name": name_t,
                    "goal_args": goal_args,
                    "maze_size_scaling": 8,
                    "random_start": random_start,
                    "maze_args": maze_args,
                },
                max_episode_steps=max_timestep,
            )

            gym.register(
                id=name_t + Test + "-v1",
                entry_point="goal_env.mujoco.create_maze_env:create_maze_env",
                kwargs={
                    "env_name": name_t,
                    "goal_args": goal_args,
                    "maze_size_scaling": 4,
                    "random_start": random_start,
                    "maze_args": maze_args,
                },
                max_episode_steps=max_timestep,
            )

            gym.register(
                id=name_t + Test + "-v2",
                entry_point="goal_env.mujoco.create_maze_env:create_maze_env",
                kwargs={
                    "env_name": name_t,
                    "goal_args": goal_args,
                    "maze_size_scaling": 2,
                    "random_start": random_start,
                    "maze_args": maze_args,
                },
                max_episode_steps=max_timestep,
            )

def register_fetch_envs():
    """Register fetch environments"""
    gym.register(
        id="Reacher3D-v0",
        entry_point="goal_env.mujoco.create_fetch_env:create_fetch_env",
        kwargs={"env_name": "Reacher3D-v0"},
        max_episode_steps=100,
    )

    gym.register(
        id="Pusher-v0",
        entry_point="goal_env.mujoco.create_fetch_env:create_fetch_env",
        kwargs={"env_name": "Pusher-v0"},
        max_episode_steps=100,
    )
    
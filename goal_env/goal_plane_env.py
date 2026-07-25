import gymnasium as gym
import copy
import numpy as np
import cv2
from collections import OrderedDict


class GoalPlane(gym.Env):
    # Modern Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_name,
        type="random",
        maze_size=16.0,
        action_size=1.0,
        distance=0.1,
        start=None,
        goals=None,
        render_mode=None,
    ):
        super(GoalPlane, self).__init__()
        
        # Store render mode
        self.render_mode = render_mode
        
        # Create the underlying environment with render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        self.maze_size = maze_size
        self.action_size = action_size

        self.action_space = gym.spaces.Box(
            low=-action_size, high=action_size, shape=(2,), dtype="float32"
        )

        self.ob_space = gym.spaces.Box(
            low=0.0, high=maze_size, shape=(2,), dtype="float32"
        )

        self.easy_goal_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.maze_size, self.maze_size / 2], dtype=np.float32),
            dtype=np.float32,
        )
        self.mid_goal_space = gym.spaces.Box(
            low=np.array([self.maze_size / 2, self.maze_size / 2], dtype=np.float32),
            high=np.array([self.maze_size, self.maze_size], dtype=np.float32),
            dtype=np.float32,
        )
        self.hard_goal_space = gym.spaces.Box(
            low=np.array([0.0, self.maze_size * 0.65], dtype=np.float32),
            high=np.array([self.maze_size / 2, self.maze_size], dtype=np.float32),
            dtype=np.float32,
        )
        self.type = type
        if self.type == "random":
            self.goal_space = self.ob_space
        elif self.type == "easy":
            self.goal_space = self.easy_goal_space
        elif self.type == "mid":
            self.goal_space = self.mid_goal_space
        elif self.type == "hard":
            self.goal_space = self.hard_goal_space

        self.distance = distance
        self.goals = goals
        self.start = start

        # Modern Gymnasium uses spaces.Dict for dict spaces
        self.observation_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "observation": self.ob_space,
                    "desired_goal": self.goal_space,
                    "achieved_goal": self.ob_space,
                }
            )
        )
        self.goal = None

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return float(reward)  # Ensure reward is Python float

    def change_mode(self, mode="mid"):
        if mode == "random":
            self.goal_space = self.ob_space
        elif mode == "easy":
            self.goal_space = self.easy_goal_space
        elif mode == "mid":
            self.goal_space = self.mid_goal_space
        elif mode == "hard":
            self.goal_space = self.hard_goal_space

    def step(self, action):
        assert self.goal is not None
        
        # Modern Gymnasium step returns 5 values
        observation, reward, terminated, truncated, info = self.env.step(
            np.array(action, dtype=np.float32) / self.maze_size
        )
        
        # Ensure observation is float32 and within bounds
        observation = np.array(observation, dtype=np.float32) * self.maze_size
        observation = np.clip(observation, 0.0, self.maze_size)

        out = {
            "observation": observation,
            "desired_goal": self.goal.astype(np.float32),
            "achieved_goal": observation,
        }
        reward = -np.linalg.norm(observation - self.goal)
        reward = float(reward)  # Ensure reward is Python float
        info["is_success"] = reward > -self.distance
        
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return out, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Modern Gymnasium reset signature
        if seed is not None:
            np.random.seed(seed)
            # Also seed the underlying environment if it supports it
            if hasattr(self.env, 'reset'):
                self.env.reset(seed=seed)
        
        if self.start is not None:
            # Reset underlying environment
            obs, info = self.env.reset()
            observation = np.array(self.start, dtype=np.float32)
            # Check if the environment has restore method
            if hasattr(self.env, 'restore'):
                self.env.restore(observation / self.maze_size)
            else:
                # If no restore method, we need to handle differently
                # This is a fallback - might not work perfectly
                pass
        else:
            obs, info = self.env.reset()
            observation = np.array(obs, dtype=np.float32)
        
        if self.goals is None:
            condition = True
            while condition:  # note: goal should not be in the block
                self.goal = self.goal_space.sample()
                # Ensure goal is float32
                self.goal = self.goal.astype(np.float32)
                # Check if environment has check_inside method
                if hasattr(self.env, 'check_inside'):
                    condition = self.env.check_inside(self.goal / self.maze_size)
                else:
                    # Fallback - assume it's valid
                    condition = False
        else:
            self.goal = np.array(self.goals, dtype=np.float32)
            
        out = {
            "observation": observation, 
            "desired_goal": self.goal,
            "achieved_goal": observation
        }
        
        # Modern Gymnasium returns (obs, info)
        return out, {}

    def render(self):
        # Modern render method without mode parameter
        # The render mode is set at environment creation
        if self.render_mode is None:
            return
        
        # Handle different render modes for the underlying environment
        if hasattr(self.env, 'render'):
            if self.render_mode == "rgb_array":
                image = self.env.render()
            else:
                # For "human" mode, render the underlying environment
                self.env.render()
                # We'll create our own overlay
                image = self.env.render() if hasattr(self.env, 'render') else None
        else:
            # Fallback if underlying env doesn't support render
            image = None
            
        if image is not None:
            goal_loc = copy.copy(self.goal)
            goal_loc[0] = goal_loc[0] / self.maze_size * image.shape[1]
            goal_loc[1] = goal_loc[1] / self.maze_size * image.shape[0]
            cv2.circle(image, (int(goal_loc[0]), int(goal_loc[1])), 10, (0, 255, 0), -1)
            
            if self.render_mode == "human":
                cv2.imshow("image", image)
                cv2.waitKey(2)
            elif self.render_mode == "rgb_array":
                return image
        elif self.render_mode == "human":
            # If we can't get an image, at least show a simple visualization
            print(f"Goal position: {self.goal}")

    def close(self):
        # Clean up resources
        if hasattr(self.env, 'close'):
            self.env.close()
        cv2.destroyAllWindows()
from .reacher import Reacher3DEnv
from .pusher import PusherEnv
from collections import OrderedDict
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.envs.registration import EnvSpec


class GoalWrapper(Wrapper):
    def __init__(
        self,
        env,
        env_name,
        reward_shaping="sparse",
        seed=0,
        subgoal_repr="subspace",
        mask_goal_in_obs=False,
        render_mode=None,
    ):
        super(GoalWrapper, self).__init__(env)
        self.env_name = env_name
        self._render_mode = render_mode
        
        ob_space = env.observation_space
        high = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        low = -high
        goal_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if subgoal_repr == "subspace":
            achieved_goal_space = goal_space
        elif subgoal_repr == "whole":
            achieved_goal_space = ob_space
        else:
            raise NotImplementedError
        self.subgoal_repr = subgoal_repr

        self.observation_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "observation": ob_space,
                    "desired_goal": goal_space,
                    "achieved_goal": achieved_goal_space,
                }
            )
        )

        self.distance_threshold = 0.25
        self.reward_shaping = reward_shaping
        self.mask_goal_in_obs = mask_goal_in_obs

    def step(self, action):
        # Modern step returns 5 values
        obs, sparse_reward, terminated, truncated, info = self.env.step(action)
        
        if self.env_name == "Reacher3D-v0":
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError

        if self.mask_goal_in_obs:
            obs[7:10] = 0.0

        out = {
            "observation": obs.astype(np.float32),
            "desired_goal": self.env.goal.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
        }

        if self.reward_shaping == "dense":
            reward = -np.sum(np.square(achieved_goal - self.env.goal))
            reward -= 0.0001 * np.square(action).sum()
            reward = float(reward)
        elif self.reward_shaping == "sparse":
            reward = float(sparse_reward)
        else:
            raise NotImplementedError

        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return out, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        if self.env_name == "Reacher3D-v0":
            achieved_goal = self.env.get_EE_pos(obs[None]).squeeze()
        elif self.env_name == "Pusher-v0":
            achieved_goal = self.env.ac_goal_pos
        else:
            raise NotImplementedError

        if self.mask_goal_in_obs:
            obs[7:10] = 0.0

        out = {
            "observation": obs.astype(np.float32),
            "desired_goal": self.env.goal.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
        }

        return out, {}
    def render(self):
        """Modern Gymnasium render method"""
        if self._render_mode == "human":
            return self.env.render()
        elif self._render_mode == "rgb_array":
            return self.env.render()
        return None


def create_fetch_env(
    env_name=None,
    seed=0,
    reward_shaping="dense",
    subgoal_repr="subspace",
    mask_goal_in_obs=False,
    render_mode=None,
):
    if env_name == "Reacher3D-v0":
        cls = Reacher3DEnv
    elif env_name == "Pusher-v0":
        cls = PusherEnv
    else:
        raise NotImplementedError

    # Create environment with render_mode
    gym_env = cls(render_mode=render_mode)
    gym_env.reset()
    
    return GoalWrapper(
        gym_env,
        env_name,
        reward_shaping=reward_shaping,
        seed=seed,
        subgoal_repr=subgoal_repr,
        mask_goal_in_obs=mask_goal_in_obs,
        render_mode=render_mode,
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
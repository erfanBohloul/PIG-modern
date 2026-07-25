"""Wrapper for creating the point environment in gym_mujoco."""

import math
import mujoco
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


class PointEnv(MujocoEnv, utils.EzPickle):
    """Modernized Point Environment for MuJoCo"""
    
    FILE = "point.xml"
    ORI_IND = 2

    def __init__(
        self, 
        file_path=None, 
        expose_all_qpos=True,
        render_mode=None,
        **kwargs
    ):
        self._expose_all_qpos = expose_all_qpos

        # Determine observation space
        if self._expose_all_qpos:
            obs_dim = 3 + 3  # qpos[:3] + qvel[:3]
        else:
            obs_dim = 1 + 3  # qpos[2:3] + qvel[:3]
        
        # Create observation space
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Modern MujocoEnv initialization with observation_space
        if file_path is not None:
            MujocoEnv.__init__(
                self, 
                file_path, 
                1, 
                observation_space=observation_space,
                render_mode=render_mode,
                **kwargs
            )
        else:
            # Use default model path if no file_path provided
            MujocoEnv.__init__(
                self, 
                self.FILE, 
                1, 
                observation_space=observation_space,
                render_mode=render_mode,
                **kwargs
            )
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        return self.model

    def step(self, action):
        """Modern Gymnasium step method"""
        action = np.array(action, dtype=np.float32)
        action[0] = 0.2 * action[0]
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        ori = qpos[2]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0] = np.clip(qpos[0] + dx, -100, 100)
        qpos[1] = np.clip(qpos[1] + dy, -100, 100)
        qvel = self.data.qvel
        self.set_state(qpos, qvel)
        for _ in range(0, self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        next_obs = self._get_obs().astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return next_obs, reward, terminated, truncated, info

    def _get_obs(self):
        if self._expose_all_qpos:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[:3],  # Only point-relevant coords.
                    self.data.qvel.flat[:3],
                ]
            )
        else:
            obs = np.concatenate([self.data.qpos.flat[2:3], self.data.qvel.flat[:3]])
        return obs.astype(np.float32)

    def reset_model(self):
        """Reset the model to initial state"""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        # Set everything other than point to original position and 0 velocity.
        qpos[3:] = self.init_qpos[3:]
        qvel[3:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, seed=None, options=None):
        """Modern Gymnasium reset method"""
        # Handle seeding
        if seed is not None:
            self.np_random, _ = utils.seeding.np_random(seed)
        
        # Call parent reset
        obs = self.reset_model()
        
        # Modern Gymnasium returns (obs, info)
        return obs.astype(np.float32), {}

    def viewer_setup(self):
        """Setup the viewer camera"""
        if self.viewer is not None:
            self.viewer.cam.distance = self.model.stat.extent

    def render(self):
        """Modern Gymnasium render method"""
        if self.render_mode == "human":
            return super().render()
        elif self.render_mode == "rgb_array":
            return super().render()
        return None

    def get_ori(self):
        return self.data.qpos[self.__class__.ORI_IND]

    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        qpos = np.copy(self.data.qpos)
        return qpos[:2]
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()
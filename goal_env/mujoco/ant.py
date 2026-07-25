# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the ant environment in gym_mujoco."""

import math
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AntEnv(MujocoEnv, utils.EzPickle):
    """Modernized Ant Environment for MuJoCo"""
    
    FILE = "ant.xml"
    ORI_IND = 3

    def __init__(
        self,
        file_path=None,
        expose_all_qpos=True,
        expose_body_coms=None,
        expose_body_comvels=None,
        render_mode=None,
        **kwargs
    ):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms if expose_body_coms is not None else []
        self._expose_body_comvels = expose_body_comvels if expose_body_comvels is not None else []
        self._body_com_indices = {}
        self._body_comvel_indices = {}

        # Determine observation space
        if self._expose_all_qpos:
            obs_dim = 15 + 14  # qpos[:15] + qvel[:14]
        else:
            obs_dim = 13 + 14  # qpos[2:15] + qvel[:14]
        
        # Add body coms
        obs_dim += len(self._expose_body_coms) * 3  # Each com is 3D
        obs_dim += len(self._expose_body_comvels) * 3  # Each comvel is 3D
        
        # Create observation space
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Modern MujocoEnv initialization with observation_space
        if file_path is not None:
            MujocoEnv.__init__(
                self, 
                file_path, 
                5, 
                observation_space=observation_space,
                render_mode=render_mode,
                **kwargs
            )
        else:
            # Use default model path if no file_path provided
            MujocoEnv.__init__(
                self, 
                self.FILE, 
                5, 
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
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        
        # Check if terminated (ant fell over)
        z_pos = self.data.qpos[2]
        terminated = bool(z_pos < 0.2)
        truncated = False
        
        ob = self._get_obs().astype(np.float32)
        info = {
            "reward_forward": float(forward_reward),
            "reward_ctrl": float(-ctrl_cost),
            "reward_survive": float(survive_reward),
        }
        
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return ob, float(reward), terminated, truncated, info

    def _get_obs(self):
        # No cfrc observation
        if self._expose_all_qpos:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[:15],  # Ensures only ant obs.
                    self.data.qvel.flat[:14],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[2:15],
                    self.data.qvel.flat[:14],
                ]
            )

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])
        
        return obs.astype(np.float32)

    def reset_model(self):
        """Reset the model to initial state"""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
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
        ori = [0, 1, 0, 0]
        rot = self.data.qpos[
            self.__class__.ORI_IND : self.__class__.ORI_IND + 4
        ]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.data.qpos[:2]
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()
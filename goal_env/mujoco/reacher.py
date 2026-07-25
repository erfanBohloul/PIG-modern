from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


class Reacher3DEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        self.viewer = None
        self.num_timesteps = 0
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3, dtype=np.float32)
        
        # Calculate obs dimension before init (we know it from the XML structure)
        # Reacher3D has: qpos (7 joints + 3 goal) = 10, qvel (7 joints - 3) = 4
        obs_dim = 7 + 3 + 4  # qpos (7 joints + 3 goal) + qvel (7 - 3)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        MujocoEnv.__init__(
            self, 
            os.path.join(dir_path, "assets/reacher3d.xml"), 
            2,
            observation_space=observation_space,
            render_mode=render_mode
        )


    def step(self, a):
        self.num_timesteps += 1
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs().astype(np.float32)
        reward_ctrl = 0.0001 * -np.square(a).sum()

        fail = True
        if np.sqrt(np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal))) <= 0.25:
            fail = False

        reward = reward_ctrl - float(fail)
        terminated = self.num_timesteps >= 100
        truncated = False
        info = {"is_success": not fail}
        
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return ob, float(reward), terminated, truncated, info

    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 2.5
            self.viewer.cam.elevation = -30
            self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.goal = qpos[-3:].astype(np.float32)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, seed=None, options=None):
        """Modern Gymnasium reset method"""
        if seed is not None:
            self.np_random, _ = utils.seeding.np_random(seed)
        self.num_timesteps = 0
        obs = self.reset_model()
        return obs.astype(np.float32), {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat[:-3],
            ]
        ).astype(np.float32)

    def render(self):
        """Modern Gymnasium render method"""
        if self.render_mode == "human":
            return super().render()
        elif self.render_mode == "rgb_array":
            return super().render()
        return None

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )

        rot_axis = np.concatenate(
            [
                np.cos(theta2) * np.cos(theta1),
                np.cos(theta2) * np.sin(theta1),
                -np.sin(theta2),
            ],
            axis=1,
        )
        rot_perp_axis = np.concatenate(
            [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
        )
        cur_end = np.concatenate(
            [
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2),
            ],
            axis=1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[
                np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
            ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(
                new_rot_perp_axis, axis=1, keepdims=True
            )
            rot_axis, rot_perp_axis, cur_end = (
                new_rot_axis,
                new_rot_perp_axis,
                cur_end + length * new_rot_axis,
            )

        return cur_end

    def close(self):
        """Close the environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()


if __name__ == "__main__":
    env = Reacher3DEnv()
    done = False
    obs, info = env.reset()
    counter = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        counter += 1
        print(obs, reward, terminated, info)
        if terminated or truncated:
            break
    print(counter)
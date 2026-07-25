from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


class PusherEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        self.num_timesteps = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))


        # Calculate observation dimension
        # qpos: 7, qvel: 7, tips_arm: 3, object: 3
        obs_dim = 7 + 7 + 3 + 3
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        MujocoEnv.__init__(
            self, 
            "%s/assets/pusher.xml" % dir_path, 
            4,
            observation_space=observation_space,
            render_mode=render_mode
        )
        utils.EzPickle.__init__(self)
        self.reset_model()

    def _get_obs_dim(self):
        """Calculate observation dimension"""
        # qpos: 7, qvel: 7, tips_arm: 3, object: 3
        return 7 + 7 + 3 + 3

    def step(self, a):
        self.num_timesteps += 1
        self.do_simulation(a, self.frame_skip)
        obj_pos = self.get_body_com("object")
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_ctrl = 0.001 * -np.square(a).sum()

        fail = True
        if np.sqrt(np.sum(np.square(vec_2))) <= 0.25:
            fail = False
        
        ob = self._get_obs().astype(np.float32)
        reward = -float(fail) + reward_ctrl
        terminated = self.num_timesteps >= 100
        truncated = False
        info = {"is_success": not fail}

        # TODO: validity check
        self.ac_goal_pos = np.concatenate(
            (self.get_body_com("object"), self.get_body_com("tips_arm"))
        )

        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return ob, float(reward), terminated, truncated, info

    def viewer_setup(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0], dtype=np.float32)
        self.cylinder_pos = np.array([-0.25, 0.15], dtype=np.float32) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        # TODO: validity check
        self.ac_goal_pos = np.concatenate(
            (self.get_body_com("object"), self.get_body_com("tips_arm"))
        )
        self.goal = np.concatenate(
            (self.get_body_com("goal"), self.get_body_com("goal"))
        )

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = utils.seeding.np_random(seed)
        self.num_timesteps = 0
        obs = self.reset_model()
        return obs.astype(np.float32), {}

    def render(self):
        """Modern Gymnasium render method"""
        if self.render_mode == "human":
            return super().render()
        elif self.render_mode == "rgb_array":
            return super().render()
        return None

    def close(self):
        """Close the environment"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()


if __name__ == "__main__":
    env = PusherEnv()
    obs, info = env.reset()
    counter = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        counter += 1
        print(obs, reward, terminated, info)
        if terminated or truncated:
            break
    print(counter)
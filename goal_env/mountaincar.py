import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class MountainCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, goal_dim=1):
        super().__init__()
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None
        self.goal_dim = goal_dim

        self.action_space = spaces.Discrete(3)
        # Modern Gymnasium uses spaces.Dict for dict spaces
        self.observation_space = spaces.Dict({
            "achieved_goal": spaces.Box(
                self.low[: self.goal_dim], self.high[: self.goal_dim], dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                self.low[: self.goal_dim], self.high[: self.goal_dim], dtype=np.float32
            ),
            "observation": spaces.Box(self.low, self.high, dtype=np.float32),
        })

        self.seed_value = None

    def get_obs(self):
        return {
            "achieved_goal": np.array(self.state)[: self.goal_dim].astype(np.float32),
            "desired_goal": np.array([self.goal_position, 0][: self.goal_dim]).astype(np.float32),
            "observation": np.array(self.state).astype(np.float32),
        }

    def seed(self, seed=None):
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        # Modern Gymnasium reset signature
        if seed is not None:
            self.seed(seed)
        elif self.seed_value is not None:
            np.random.seed(self.seed_value)
            
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        # Return obs and info dict
        return self.get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        info = {"is_success": False}
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position)
        reward = -1.0
        if terminated:
            reward = 0.0
            info["is_success"] = True

        self.state = (position, velocity)
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return self.get_obs(), reward, terminated, False, info

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        # Modern Gymnasium render method doesn't take mode parameter
        # The render mode is set when creating the environment
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=self.render_mode == "rgb_array")

    def get_keys_to_action(self):
        # control with left and right arrow keys
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def compute_reward(self, state, goal):
        """
        to be finish
        :param state:
        :param goal:
        :return:
        """
        pass

    def get_pairwise(self, state, target):
        """
        to be finish
        :param state:
        :param target:
        :return:
        """
        pass
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


def line_intersection(line1, line2):
    # calculate the intersection point
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def check_cross(x0, y0, x1, y1):
    """Check if segments cross by computing cross products"""
    # Convert to numpy arrays
    x0 = np.array(x0, dtype=np.float64)
    y0 = np.array(y0, dtype=np.float64)
    x1 = np.array(x1, dtype=np.float64)
    y1 = np.array(y1, dtype=np.float64)
    
    # Compute cross products for 2D vectors
    # For 2D vectors, cross product is scalar: x1*y2 - y1*x2
    cross1 = np.cross(x1 - x0, y0 - x0)
    cross2 = np.cross(y0 - x0, y1 - x0)
    
    # If cross1 or cross2 is a scalar, return as is
    # If they are arrays with shape (2,), convert to scalar
    if isinstance(cross1, np.ndarray) and cross1.shape == (2,):
        cross1 = cross1[0] * cross1[1]  # This shouldn't happen
    if isinstance(cross2, np.ndarray) and cross2.shape == (2,):
        cross2 = cross2[0] * cross2[1]  # This shouldn't happen
    
    return cross1, cross2


def check_itersection(x0, y0, x1, y1):
    EPS = 1e-10

    def sign(x):
        if x > EPS:
            return 1
        if x < -EPS:
            return -1
        return 0

    try:
        f1, f2 = check_cross(x0, y0, x1, y1)
        f3, f4 = check_cross(x1, y1, x0, y0)
        
        # Handle case where f values might be arrays
        f1 = f1 if isinstance(f1, (int, float)) else f1.item() if hasattr(f1, 'item') else 0
        f2 = f2 if isinstance(f2, (int, float)) else f2.item() if hasattr(f2, 'item') else 0
        f3 = f3 if isinstance(f3, (int, float)) else f3.item() if hasattr(f3, 'item') else 0
        f4 = f4 if isinstance(f4, (int, float)) else f4.item() if hasattr(f4, 'item') else 0
        
        if (
            sign(f1) == sign(f2)
            and sign(f3) == sign(f4)
            and sign(f1) != 0
            and sign(f3) != 0
        ):
            return True
        return False
    except:
        return False


class PlaneBase(gym.Env):
    # Modern Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, rects, R, is_render=False, size=512, render_mode=None):
        super().__init__()
        self.rects = rects
        self.n = len(self.rects)
        self.size = size
        self.map = np.ones((size, size, 3), dtype=np.uint8) * 255
        self.R = R
        self.R2 = R ** 2
        self.board = np.array([[0, 0], [1, 1]], dtype="float32")
        
        # Store render mode
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-R, high=R, shape=(2,), dtype="float32")
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype="float32"
        )

        # Initialize rendering
        self.viewer = None
        if is_render or render_mode is not None:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            self.image_name = "image"

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Fix: Check intersection of rectangles properly
                if check_itersection(
                    self.rects[i][0],
                    self.rects[i][1],
                    self.rects[j][0],
                    self.rects[j][1],
                ):
                    raise Exception("Rectangle interaction with each other")

        for ((x0, y0), (x1, y1)) in rects:
            x0, y0 = int(x0 * size), int(y0 * size)
            x1, y1 = int(x1 * size), int(y1 * size)
            cv2.rectangle(self.map, (x0, y0), (x1, y1), (0, 255, 0), 1)

            ps = np.array(
                [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(self.map, ps, (127, 127, 127))

        self.state = (0, 0)
        self.reset()

    def restore(self, obs):
        self.state = (float(obs[0]), float(obs[1]))

    def rect_lines(self, rect):
        (x0, y0), (x1, y1) = rect
        yield (x0, y0), (x1, y0)
        yield (x1, y0), (x1, y1)
        yield (x1, y1), (x0, y1)
        yield (x0, y1), (x0, y0)

    def l2dist(self, x, y):
        return ((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2)

    def check_inside(self, p):
        EPS = 1e-10
        for i in self.rects:
            if (
                p[0] > i[0][0] + EPS
                and p[0] < i[1][0] - EPS
                and p[1] > i[0][1] + EPS
                and p[1] < i[1][1] - EPS
            ):
                return True
        return False

    def reset(self, seed=None, options=None):
        # Modern Gymnasium reset signature
        if seed is not None:
            np.random.seed(seed)
            
        inside_rect = True
        while inside_rect:
            a, b = np.random.random(), np.random.random()
            inside_rect = self.check_inside((a, b))
        self.state = (a, b)
        # Return obs and info dict
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        dx, dy = action
        l = 0.0001
        p = (self.state[0] + dx * l, self.state[1] + dy * l)
        if self.check_inside(p) or p[0] > 1 or p[1] > 1 or p[0] < 0 or p[1] < 0:
            # Convert reward to Python float
            return np.array(self.state, dtype=np.float32), 0.0, False, False, {}

        dest = (self.state[0] + dx, self.state[1] + dy)

        md = self.l2dist(self.state, dest)

        _dest = dest
        line = (self.state, dest)

        for i in list(self.rects) + [self.board]:
            for l in self.rect_lines(i):
                try:
                    if check_itersection(self.state, dest, l[0], l[1]):
                        inter_point = line_intersection(line, l)
                        d = self.l2dist(self.state, inter_point)
                        if d < md:
                            md = d
                            _dest = inter_point
                except:
                    continue

        self.restore(_dest)
        # Convert reward to Python float
        reward = float(-md)
        # Modern Gymnasium: (obs, reward, terminated, truncated, info)
        return np.array(self.state, dtype=np.float32), reward, False, False, {}

    def render(self):
        # Modern render method without mode parameter
        if self.render_mode is None:
            return
            
        image = self.map.copy()
        x, y = self.state
        x = int(x * self.size)
        y = int(y * self.size)
        cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
        
        if self.render_mode == "human":
            cv2.imshow("image", image)
            cv2.waitKey(2)
        elif self.render_mode == "rgb_array":
            return image

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            cv2.destroyWindow("image")
            self.viewer = None


class NaivePlane(PlaneBase):
    def __init__(self, is_render=False, R=300, size=512, render_mode=None):
        super().__init__(
            [
                np.array([[128, 128], [300, 386]]) / 512,
                np.array([[400, 400], [500, 500]]) / 512,
            ],
            R,
            is_render=is_render,
            size=size,
            render_mode=render_mode,
        )


class NaivePlane2(PlaneBase):
    # two rectangle
    def __init__(self, is_render=False, R=300, size=512, render_mode=None):
        super().__init__(
            [
                np.array([[64, 64], [256, 256]]) / 512,
                np.array([[300, 128], [400, 500]]) / 512,
            ],
            R,
            is_render=is_render,
            size=size,
            render_mode=render_mode,
        )


class NaivePlane3(PlaneBase):
    # four rectangle
    def __init__(self, is_render=False, R=300, size=512, render_mode=None):
        super().__init__(
            [
                np.array([[64, 64], [192, 192]]) / 512,
                np.array([[320, 64], [448, 192]]) / 512,
                np.array([[320, 320], [448, 448]]) / 512,
                np.array([[64, 320], [192, 448]]) / 512,
            ],
            R,
            is_render=is_render,
            size=size,
            render_mode=render_mode,
        )


class NaivePlane4(PlaneBase):
    # four rectangle
    def __init__(self, is_render=False, R=300, size=512, render_mode=None):
        super().__init__(
            [
                np.array([[64, 64], [192, 512]]) / 512,
                np.array([[320, 64], [448, 512]]) / 512,
            ],
            R,
            is_render=is_render,
            size=size,
            render_mode=render_mode,
        )


class NaivePlane5(PlaneBase):
    # four rectangle
    def __init__(self, is_render=False, R=300, size=512, render_mode=None):
        super().__init__(
            [
                np.array([[0, 1.0 / 3], [2.0 / 3, 2.0 / 3]]),
            ],
            R,
            is_render=is_render,
            size=size,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    # Test the environment
    env = NaivePlane5(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: obs={obs}, reward={reward:.4f}, type={type(reward)}")
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
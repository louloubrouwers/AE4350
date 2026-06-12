"""
drone_env.py
2D continuous drone-navigation environment for AE4350.

The drone is a point mass with continuous acceleration control in (ax, ay).
The world contains rectangular no-fly zones (axis-aligned boxes) and stochastic wind.
The agent observes its own state plus ray-cast distances to the nearest obstacle
or wall along evenly spaced directions, mimicking simple insect-like sensing.

State:
    [x, y, vx, vy, dx_goal, dy_goal, r_1, r_2, ..., r_N]

    x, y:
        drone position

    vx, vy:
        drone velocity

    dx_goal, dy_goal:
        goal position relative to the drone

    r_i:
        ray distances clipped to ray_max_dist

Action:
    continuous action in [-1, 1]^2:
        a = (ax_norm, ay_norm)

    This is internally scaled to:
        acceleration = action * max_accel

Reward:
    reward =
        progress reward
        + step penalty
        + action penalty
        + proximity penalty near walls/obstacles
        + terminal goal bonus
        + terminal collision penalty

Termination:
    terminated = reached goal OR collision with obstacle/wall
    truncated  = time limit reached

This file is intentionally framework-agnostic on the RL side.
It only depends on numpy and gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "This environment requires gymnasium. Install with:\n"
        "    pip install gymnasium"
    ) from e


# ---------------------------------------------------------------------------
# Obstacles and configuration
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """Axis-aligned rectangular no-fly zone."""

    x: float  # bottom-left corner x
    y: float  # bottom-left corner y
    w: float  # width along +x
    h: float  # height along +y

    def overlaps_circle(self, px: float, py: float, radius: float) -> bool:
        """
        Return True if a disk of given radius centered at (px, py)
        intersects the rectangle.
        """

        cx = float(np.clip(px, self.x, self.x + self.w))
        cy = float(np.clip(py, self.y, self.y + self.h))

        return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2


@dataclass
class EnvConfig:
    # World
    world_size: float = 20.0
    dt: float = 0.05
    max_steps: int = 400

    # Drone dynamics
    max_accel: float = 2.0
    max_speed: float = 4.0
    drone_radius: float = 0.3
    goal_radius: float = 0.6

    # Wind disturbance, added directly to acceleration
    wind_mean: Tuple[float, float] = (0.0, 0.0)
    wind_std: float = 0.5

    # Sensing
    n_rays: int = 8
    ray_max_dist: float = 8.0

    # Reward weights
    r_goal: float = 200.0
    r_collision: float = -75.0
    r_step: float = -0.05
    r_action: float = -0.01
    r_progress: float = 1.0

    # Smooth danger penalty near walls/obstacles.
    # This helps PPO learn obstacle avoidance before actual collision.
    r_proximity: float = -1.0
    proximity_threshold: float = 2.0

    # Fixed spawn / goal.
    # None means random sampling.
    fixed_start: Optional[Tuple[float, float]] = None
    fixed_goal: Optional[Tuple[float, float]] = None
    min_start_goal_dist: float = 5.0

    # Curriculum / controlled randomization.
    # If jitter_start_center and jitter_goal_center are set, reset() samples
    # start/goal in a box around those centers instead of the full world.
    jitter_start_center: Optional[Tuple[float, float]] = None
    jitter_goal_center: Optional[Tuple[float, float]] = None
    jitter_radius: float = 0.0

    # Obstacles
    obstacles: List[Obstacle] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Difficulty configurations
# ---------------------------------------------------------------------------

def easy_obstacles() -> List[Obstacle]:
    """
    Three obstacles in a 20x20 world.
    This is the main small-world obstacle layout.
    """

    return [
        Obstacle(6.5, 4.5, 1.8, 5.5),
        Obstacle(12.0, 8.5, 1.8, 5.5),
        Obstacle(4.0, 14.5, 4.0, 1.2),
    ]


def medium_obstacles() -> List[Obstacle]:
    """
    Eight obstacles in a 30x30 world.
    Used later for generalization/evaluation.
    """

    return [
        Obstacle(5.0, 5.0, 3.0, 4.0),
        Obstacle(12.0, 3.0, 2.0, 8.0),
        Obstacle(18.0, 10.0, 6.0, 2.0),
        Obstacle(8.0, 15.0, 4.0, 3.0),
        Obstacle(20.0, 20.0, 3.0, 5.0),
        Obstacle(14.0, 22.0, 4.0, 2.0),
        Obstacle(3.0, 22.0, 4.0, 4.0),
        Obstacle(25.0, 5.0, 3.0, 10.0),
    ]


def hard_obstacles() -> List[Obstacle]:
    """
    Fifteen obstacles in a 50x50 world arranged like a city grid.
    This is mainly for later stress testing.
    """

    return [
        # Row 1
        Obstacle(2.0, 2.0, 7.0, 7.0),
        Obstacle(15.0, 2.0, 9.0, 5.0),
        Obstacle(28.0, 2.0, 7.0, 8.0),
        Obstacle(41.0, 2.0, 7.0, 6.0),

        # Row 2
        Obstacle(2.0, 14.0, 8.0, 7.0),
        Obstacle(15.0, 14.0, 7.0, 8.0),
        Obstacle(28.0, 14.0, 8.0, 6.0),
        Obstacle(41.0, 14.0, 6.0, 8.0),

        # Row 3, central plaza missing
        Obstacle(2.0, 26.0, 7.0, 8.0),
        Obstacle(15.0, 26.0, 8.0, 7.0),
        Obstacle(41.0, 26.0, 7.0, 7.0),

        # Row 4
        Obstacle(2.0, 38.0, 6.0, 8.0),
        Obstacle(15.0, 38.0, 8.0, 9.0),
        Obstacle(28.0, 38.0, 7.0, 9.0),
        Obstacle(41.0, 38.0, 7.0, 9.0),
    ]


def easy_config() -> EnvConfig:
    """
    Small 20x20 world, three obstacles, mild wind.
    This is the first full environment after curriculum.
    """

    return EnvConfig(
        world_size=20.0,
        max_steps=400,
        max_accel=2.0,
        max_speed=4.0,
        wind_mean=(0.0, 0.0),
        wind_std=0.3,
        n_rays=8,
        ray_max_dist=8.0,
        r_goal=200.0,
        r_collision=-75.0,
        r_step=-0.05,
        r_action=-0.01,
        r_progress=1.0,
        r_proximity=-1.0,
        proximity_threshold=2.0,
        min_start_goal_dist=5.0,
        obstacles=easy_obstacles(),
    )


def medium_config() -> EnvConfig:
    """
    30x30 world, eight obstacles, moderate wind.
    """

    return EnvConfig(
        world_size=30.0,
        max_steps=600,
        max_accel=2.5,
        max_speed=5.0,
        wind_mean=(0.3, 0.0),
        wind_std=0.7,
        n_rays=8,
        ray_max_dist=10.0,
        r_goal=200.0,
        r_collision=-75.0,
        r_step=-0.05,
        r_action=-0.01,
        r_progress=1.0,
        r_proximity=-1.0,
        proximity_threshold=2.0,
        min_start_goal_dist=8.0,
        obstacles=medium_obstacles(),
    )


def hard_config() -> EnvConfig:
    """
    50x50 world, fifteen obstacles, stronger wind.
    """

    return EnvConfig(
        world_size=50.0,
        max_steps=1000,
        max_accel=3.0,
        max_speed=6.0,
        wind_mean=(0.5, -0.2),
        wind_std=0.8,
        n_rays=12,
        ray_max_dist=12.0,
        r_goal=200.0,
        r_collision=-75.0,
        r_step=-0.05,
        r_action=-0.01,
        r_progress=1.0,
        r_proximity=-1.0,
        proximity_threshold=2.5,
        min_start_goal_dist=15.0,
        obstacles=hard_obstacles(),
    )


def default_obstacles() -> List[Obstacle]:
    """
    Backwards-compatible alias.
    """

    return easy_obstacles()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DroneNavEnv(gym.Env):
    """
    Continuous 2D drone navigation with wind and rectangular no-fly zones.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if config is None:
            config = easy_config()

        self.cfg = config
        self.render_mode = render_mode

        # Observation: [x, y, vx, vy, dx_goal, dy_goal, rays...]
        ws = self.cfg.world_size
        ms = self.cfg.max_speed
        rd = self.cfg.ray_max_dist
        n = self.cfg.n_rays

        low_state = np.concatenate(
            [
                np.array([0.0, 0.0, -ms, -ms, -ws, -ws], dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            ]
        )

        high_state = np.concatenate(
            [
                np.array([ws, ws, ms, ms, ws, ws], dtype=np.float32),
                np.full(n, rd, dtype=np.float32),
            ]
        )

        self.observation_space = spaces.Box(
            low=low_state,
            high=high_state,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Runtime state
        self.pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal: np.ndarray = np.zeros(2, dtype=np.float32)
        self.steps: int = 0
        self.prev_dist: float = 0.0
        self.trajectory: List[Tuple[float, float]] = []
        self.last_wind: np.ndarray = np.zeros(2, dtype=np.float32)
        self._renderer = None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _in_bounds(self, p: np.ndarray) -> bool:
        """
        Return True if drone disk is fully inside world boundaries.
        """

        r = self.cfg.drone_radius
        W = self.cfg.world_size

        return (
            p[0] >= r
            and p[0] <= W - r
            and p[1] >= r
            and p[1] <= W - r
        )

    def _in_any_obstacle(self, p: np.ndarray) -> bool:
        """
        Return True if the drone overlaps any rectangular obstacle.
        """

        for ob in self.cfg.obstacles:
            if ob.overlaps_circle(
                float(p[0]),
                float(p[1]),
                self.cfg.drone_radius,
            ):
                return True

        return False

    def _sample_free_point(self) -> np.ndarray:
        """
        Reject-sample a position inside the world that is not in any obstacle.
        """

        rng = self.np_random
        margin = max(self.cfg.drone_radius + 0.2, 2.0)

        for _ in range(1000):
            p = rng.uniform(
                margin,
                self.cfg.world_size - margin,
                size=2,
            ).astype(np.float32)

            if self._in_bounds(p) and not self._in_any_obstacle(p):
                return p

        # Fallback corner
        return np.array([margin, margin], dtype=np.float32)

    def _sample_jittered_point(
        self,
        center: Tuple[float, float],
        radius: float,
    ) -> np.ndarray:
        """
        Sample a free point near a given center for curriculum learning.
        This makes the jump from fixed start/goal to fully random start/goal
        much smaller, which is critical for reliable training.
        """

        rng = self.np_random
        c = np.array(center, dtype=np.float32)
        radius = float(max(radius, 0.0))

        for _ in range(1000):
            p = c + rng.uniform(-radius, radius, size=2).astype(np.float32)
            p = np.clip(
                p,
                self.cfg.drone_radius + 0.2,
                self.cfg.world_size - self.cfg.drone_radius - 0.2,
            ).astype(np.float32)

            if self._in_bounds(p) and not self._in_any_obstacle(p):
                return p

        # If jitter sampling fails, fall back to any free point.
        return self._sample_free_point()

    @staticmethod
    def _ray_vs_aabb(
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        ob: Obstacle,
    ) -> float:
        """
        Distance along ray from origin (ox, oy), direction (dx, dy),
        to first intersection with axis-aligned box.

        Returns +inf if no hit ahead.
        """

        big = float("inf")
        tmin = -big
        tmax = big

        for o, d, lo, hi in (
            (ox, dx, ob.x, ob.x + ob.w),
            (oy, dy, ob.y, ob.y + ob.h),
        ):
            if abs(d) < 1e-9:
                if o < lo or o > hi:
                    return big
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d

                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return big

        if tmax < 0:
            return big

        return max(0.0, tmin)

    def _cast_rays(self) -> np.ndarray:
        """
        Cast n_rays evenly spaced rays and return distances to nearest
        wall or obstacle.
        """

        n = self.cfg.n_rays
        rd_max = self.cfg.ray_max_dist

        ox = float(self.pos[0])
        oy = float(self.pos[1])
        W = self.cfg.world_size

        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        dists = np.full(n, rd_max, dtype=np.float32)

        for i, angle in enumerate(angles):
            dx = float(np.cos(angle))
            dy = float(np.sin(angle))

            best = rd_max

            # Distance to world walls
            if dx > 1e-9:
                best = min(best, (W - ox) / dx)
            elif dx < -1e-9:
                best = min(best, (0.0 - ox) / dx)

            if dy > 1e-9:
                best = min(best, (W - oy) / dy)
            elif dy < -1e-9:
                best = min(best, (0.0 - oy) / dy)

            # Distance to obstacles
            for ob in self.cfg.obstacles:
                t = self._ray_vs_aabb(ox, oy, dx, dy, ob)

                if t < best:
                    best = t

            dists[i] = max(0.0, min(best, rd_max))

        return dists

    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector.
        """

        rel_goal = self.goal - self.pos
        rays = self._cast_rays()

        return np.concatenate(
            [
                self.pos,
                self.vel,
                rel_goal,
                rays,
            ]
        ).astype(np.float32)

    def _get_info(self) -> dict:
        """
        Return diagnostic information.
        """

        return {
            "position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "goal": self.goal.copy(),
            "dist_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "steps": self.steps,
            "wind": self.last_wind.copy(),
        }

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Fixed start/goal has priority. Otherwise, allow a curriculum mode
        # where start and goal are random but only near known feasible centers.
        if self.cfg.fixed_start is not None:
            self.pos = np.array(self.cfg.fixed_start, dtype=np.float32)
        elif self.cfg.jitter_start_center is not None:
            self.pos = self._sample_jittered_point(
                self.cfg.jitter_start_center,
                self.cfg.jitter_radius,
            )
        else:
            self.pos = self._sample_free_point()

        if self.cfg.fixed_goal is not None:
            self.goal = np.array(self.cfg.fixed_goal, dtype=np.float32)
        elif self.cfg.jitter_goal_center is not None:
            self.goal = self._sample_jittered_point(
                self.cfg.jitter_goal_center,
                self.cfg.jitter_radius,
            )
        else:
            self.goal = self._sample_free_point()

        tries = 0
        while (
            np.linalg.norm(self.goal - self.pos) < self.cfg.min_start_goal_dist
            and tries < 100
        ):
            if self.cfg.fixed_goal is not None:
                break
            if self.cfg.jitter_goal_center is not None:
                self.goal = self._sample_jittered_point(
                    self.cfg.jitter_goal_center,
                    self.cfg.jitter_radius,
                )
            else:
                self.goal = self._sample_free_point()
            tries += 1

        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        self.trajectory = [tuple(self.pos.tolist())]
        self.last_wind = np.zeros(2, dtype=np.float32)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        accel = action * self.cfg.max_accel

        # Wind disturbance
        wind = np.array(self.cfg.wind_mean, dtype=np.float32)

        if self.cfg.wind_std > 0.0:
            wind = wind + self.np_random.normal(
                0.0,
                self.cfg.wind_std,
                size=2,
            ).astype(np.float32)

        self.last_wind = wind.copy()

        # Semi-implicit Euler integration
        self.vel = self.vel + (accel + wind) * self.cfg.dt

        speed = float(np.linalg.norm(self.vel))

        if speed > self.cfg.max_speed:
            self.vel = self.vel * (self.cfg.max_speed / speed)

        proposed_pos = self.pos + self.vel * self.cfg.dt

        # Check wall collision before clipping.
        wall_collision = not self._in_bounds(proposed_pos)

        # Clip for numerical safety and plotting.
        self.pos = np.clip(
            proposed_pos,
            0.0,
            self.cfg.world_size,
        ).astype(np.float32)

        self.trajectory.append(tuple(self.pos.tolist()))
        self.steps += 1

        # Distance to goal after movement
        dist = float(np.linalg.norm(self.goal - self.pos))

        # Base reward: progress + time cost + control effort cost
        reward = (
            self.cfg.r_step
            + self.cfg.r_progress * (self.prev_dist - dist)
            + self.cfg.r_action * float(np.sum(action ** 2))
        )

        # Extra shaping for continuous control:
        # 1) reward velocity component in the goal direction,
        # 2) penalize overspeeding close to the goal so the drone learns to brake.
        if dist > 1e-6:
            goal_dir = (self.goal - self.pos) / dist
            speed_toward_goal = float(np.dot(self.vel, goal_dir))
            reward += 0.05 * speed_toward_goal

        speed_after_clip = float(np.linalg.norm(self.vel))
        if dist < 2.0:
            reward -= 0.05 * speed_after_clip

        # Smooth proximity penalty.
        # This teaches obstacle/wall avoidance before actual collision.
        rays_now = self._cast_rays()
        min_ray = float(np.min(rays_now))

        if min_ray < self.cfg.proximity_threshold:
            danger = (
                self.cfg.proximity_threshold - min_ray
            ) / self.cfg.proximity_threshold

            reward += self.cfg.r_proximity * danger

        terminated = False
        truncated = False

        info = self._get_info()
        info["reached_goal"] = False
        info["collision"] = False
        info["wall_collision"] = False
        info["obstacle_collision"] = False
        info["min_ray"] = min_ray

        obstacle_collision = self._in_any_obstacle(self.pos)

        if wall_collision:
            reward += self.cfg.r_collision
            terminated = True
            info["collision"] = True
            info["wall_collision"] = True

        elif obstacle_collision:
            reward += self.cfg.r_collision
            terminated = True
            info["collision"] = True
            info["obstacle_collision"] = True

        elif dist <= self.cfg.goal_radius:
            reward += self.cfg.r_goal
            terminated = True
            info["reached_goal"] = True

        elif self.steps >= self.cfg.max_steps:
            truncated = True

        self.prev_dist = dist

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """
        Rendering will be handled later by the PyGame demo.
        Keep this stub so headless training remains simple.
        """

        if self.render_mode is None:
            return None

        raise NotImplementedError(
            "Rendering will be added in the PyGame demo. "
            "For training, use render_mode=None."
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = DroneNavEnv(config=easy_config())

    obs, info = env.reset(seed=0)

    print("Observation shape:", obs.shape)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial info:", info)

    total_reward = 0.0

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"t={t:02d} | reward={reward:+.3f} | "
            f"dist={info['dist_to_goal']:.2f} | "
            f"min_ray={info['min_ray']:.2f} | "
            f"collision={info['collision']} | "
            f"goal={info['reached_goal']}"
        )

        if terminated or truncated:
            break

    print("Smoke test total reward:", total_reward)
    env.close()
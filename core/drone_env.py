"""Gymnasium environment for the 2D drone navigation tasks."""

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


@dataclass
class Obstacle:
    """Axis-aligned rectangular no-fly zone."""

    x: float
    y: float
    w: float
    h: float

    def overlaps_circle(self, px: float, py: float, radius: float) -> bool:
        """Return True if a disk centered at (px, py) intersects the rectangle."""
        cx = float(np.clip(px, self.x, self.x + self.w))
        cy = float(np.clip(py, self.y, self.y + self.h))
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2


@dataclass
class EnvConfig:
    """
    Generic environment configuration.
    Final task builders and reward values are defined in env_configs.py.
    """

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
    wind_std: float = 0.0

    # Sensing
    n_rays: int = 16
    ray_max_dist: float = 10.0

    # Reward weights. Final values are applied by env_configs.py.
    r_goal: float = 350.0
    r_collision: float = -100.0
    r_step: float = -0.01
    r_action: float = -0.001
    r_progress: float = 4.0
    r_proximity: float = -0.01
    proximity_threshold: float = 1.0

    # Fixed spawn / goal. None means random sampling.
    fixed_start: Optional[Tuple[float, float]] = None
    fixed_goal: Optional[Tuple[float, float]] = None
    min_start_goal_dist: float = 5.0

    # Curriculum / controlled randomization.
    jitter_start_center: Optional[Tuple[float, float]] = None
    jitter_goal_center: Optional[Tuple[float, float]] = None
    jitter_radius: float = 0.0

    # Obstacles
    obstacles: List[Obstacle] = field(default_factory=list)


class DroneNavEnv(gym.Env):
    """Continuous 2D drone navigation with wind and rectangular obstacles."""

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

        self.cfg = config if config is not None else EnvConfig()
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

        self.pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal: np.ndarray = np.zeros(2, dtype=np.float32)
        self.steps: int = 0
        self.prev_dist: float = 0.0
        self.trajectory: List[Tuple[float, float]] = []
        self.last_wind: np.ndarray = np.zeros(2, dtype=np.float32)
        self._renderer = None

    def _in_bounds(self, p: np.ndarray) -> bool:
        r = self.cfg.drone_radius
        W = self.cfg.world_size
        return p[0] >= r and p[0] <= W - r and p[1] >= r and p[1] <= W - r

    def _in_any_obstacle(self, p: np.ndarray) -> bool:
        for ob in self.cfg.obstacles:
            if ob.overlaps_circle(float(p[0]), float(p[1]), self.cfg.drone_radius):
                return True
        return False

    def _sample_free_point(self) -> np.ndarray:
        rng = self.np_random
        margin = max(self.cfg.drone_radius + 0.2, 2.0)

        for _ in range(1000):
            p = rng.uniform(margin, self.cfg.world_size - margin, size=2).astype(np.float32)
            if self._in_bounds(p) and not self._in_any_obstacle(p):
                return p

        return np.array([margin, margin], dtype=np.float32)

    def _sample_jittered_point(
        self,
        center: Tuple[float, float],
        radius: float,
    ) -> np.ndarray:
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

        return self._sample_free_point()

    @staticmethod
    def _ray_vs_aabb(
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        ob: Obstacle,
    ) -> float:
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

            if dx > 1e-9:
                best = min(best, (W - ox) / dx)
            elif dx < -1e-9:
                best = min(best, (0.0 - ox) / dx)

            if dy > 1e-9:
                best = min(best, (W - oy) / dy)
            elif dy < -1e-9:
                best = min(best, (0.0 - oy) / dy)

            for ob in self.cfg.obstacles:
                best = min(best, self._ray_vs_aabb(ox, oy, dx, dy, ob))

            dists[i] = max(0.0, min(best, rd_max))

        return dists

    def _get_obs(self) -> np.ndarray:
        rel_goal = self.goal - self.pos
        rays = self._cast_rays()
        return np.concatenate([self.pos, self.vel, rel_goal, rays]).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "goal": self.goal.copy(),
            "dist_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "steps": self.steps,
            "wind": self.last_wind.copy(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

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
        while np.linalg.norm(self.goal - self.pos) < self.cfg.min_start_goal_dist and tries < 100:
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

        wind = np.array(self.cfg.wind_mean, dtype=np.float32)
        if self.cfg.wind_std > 0.0:
            wind = wind + self.np_random.normal(0.0, self.cfg.wind_std, size=2).astype(np.float32)

        self.last_wind = wind.copy()

        self.vel = self.vel + (accel + wind) * self.cfg.dt

        speed = float(np.linalg.norm(self.vel))
        if speed > self.cfg.max_speed:
            self.vel = self.vel * (self.cfg.max_speed / speed)

        proposed_pos = self.pos + self.vel * self.cfg.dt

        wall_collision = not self._in_bounds(proposed_pos)

        self.pos = np.clip(proposed_pos, 0.0, self.cfg.world_size).astype(np.float32)
        self.trajectory.append(tuple(self.pos.tolist()))
        self.steps += 1

        dist = float(np.linalg.norm(self.goal - self.pos))

        reward = (
            self.cfg.r_step
            + self.cfg.r_progress * (self.prev_dist - dist)
            + self.cfg.r_action * float(np.sum(action ** 2))
        )

        if dist > 1e-6:
            goal_dir = (self.goal - self.pos) / dist
            speed_toward_goal = float(np.dot(self.vel, goal_dir))
            reward += 0.05 * speed_toward_goal

        speed_after_clip = float(np.linalg.norm(self.vel))
        if dist < 2.0:
            reward -= 0.05 * speed_after_clip

        rays_now = self._cast_rays()
        min_ray = float(np.min(rays_now))

        if min_ray < self.cfg.proximity_threshold:
            danger = (self.cfg.proximity_threshold - min_ray) / self.cfg.proximity_threshold
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
        if self.render_mode is None:
            return None

        raise NotImplementedError(
            "Rendering is handled by the Pygame demo. "
            "For training, use render_mode=None."
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = DroneNavEnv(config=EnvConfig())
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

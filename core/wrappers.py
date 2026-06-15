"""
wrappers.py
Shared wrappers and environment characteristics used by all algorithms.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from core.drone_env import DroneNavEnv
from core.env_configs import make_config


DIRECTIONAL_AVOIDANCE_SETTINGS = {
    "threshold": 2.0,
    "away_reward": 1.5,
    "toward_penalty": 2.0,
    "speed_reward": 0.03,
    "min_speed_for_bonus": 0.15,
}


class DirectionalAvoidanceWrapper(gym.Wrapper):
    """
    Adds directional obstacle-avoidance shaping using the ray sensors.

    If the drone is close to the nearest sensed obstacle/wall: 
    moving toward danger is penalize and moving away from danger is rewarded.
    This wrapper is used for SAC, SB3 PPO, and custom PPO.
    """

    def __init__(
        self,
        env: gym.Env,
        threshold: float = 2.0,
        away_reward: float = 1.5,
        toward_penalty: float = 2.0,
        speed_reward: float = 0.03,
        min_speed_for_bonus: float = 0.15,
    ):
        super().__init__(env)
        self.threshold = threshold
        self.away_reward = away_reward
        self.toward_penalty = toward_penalty
        self.speed_reward = speed_reward
        self.min_speed_for_bonus = min_speed_for_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx, vy = float(obs[2]), float(obs[3])
        velocity = np.array([vx, vy], dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        rays = np.asarray(obs[6:], dtype=np.float32)

        if len(rays) > 0:
            min_idx = int(np.argmin(rays))
            min_ray = float(rays[min_idx])
            n_rays = len(rays)
            angle = 2.0 * np.pi * min_idx / n_rays
            ray_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

            if min_ray < self.threshold:
                danger = (self.threshold - min_ray) / self.threshold
                approach_speed = float(np.dot(velocity, ray_dir))

                if approach_speed > 0.0:
                    reward -= self.toward_penalty * danger * approach_speed
                else:
                    reward += self.away_reward * danger * (-approach_speed)

            info["min_ray"] = min_ray

        if speed > self.min_speed_for_bonus:
            reward += self.speed_reward * min(speed, 1.0)

        return obs, float(reward), terminated, truncated, info


def make_env(task: str, seed: Optional[int] = None, monitor: bool = False):
    env = DroneNavEnv(config=make_config(task))
    env = DirectionalAvoidanceWrapper(env, **DIRECTIONAL_AVOIDANCE_SETTINGS)

    if monitor:
        try:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        except ImportError:
            pass

    if seed is not None:
        env.reset(seed=seed)

    return env


def unwrap_drone_env(env):
    """
    Return the underlying DroneNavEnv from a possibly wrapped environment.

    Stable-Baselines3 wraps environments in objects such as VecNormalize,
    DummyVecEnv, Monitor, and custom Gym wrappers. This helper walks through
    those layers so evaluation code can access DroneNavEnv-specific fields
    such as position, goal, obstacles, and trajectory.
    """
    current = env

    # VecNormalize -> DummyVecEnv
    if hasattr(current, "venv"):
        current = current.venv

    # DummyVecEnv -> first env
    if hasattr(current, "envs"):
        current = current.envs[0]

    for _ in range(20):
        if isinstance(current, DroneNavEnv):
            return current
        if hasattr(current, "env"):
            current = current.env
        else:
            break

    return current

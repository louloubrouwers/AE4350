"""
env_configs.py
defines all tasks and parameters 

Clean task naming used in the final version:
    free_fixed
    free_jitter_r2
    free_jitter_r5
    free_jitter_r8
    free_random
    one_obstacle_fixed
    one_obstacle_jitter_r2
    one_obstacle_jitter_r4
    one_obstacle_jitter_r6
    one_obstacle_random
    two_obstacles_random
    three_obstacles_random
    three_obstacles_wind

Training, evaluation, demo, and sensitivity scripts should use these task names only.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Dict, List

from core.drone_env import EnvConfig, Obstacle


# ---------------------------------------------------------------------------
# Global final reward setting
# ---------------------------------------------------------------------------

REWARD_SETTINGS = {
    "r_goal": 350.0,
    "r_collision": -100.0,
    "r_step": -0.01,
    "r_action": -0.001,
    "r_progress": 4.0,
    "r_proximity": -0.01,
    "proximity_threshold": 1.0,
}


def apply_final_rewards(cfg: EnvConfig) -> EnvConfig:
    """Apply the reward values"""
    settings = dict(REWARD_SETTINGS)

    for key in list(settings.keys()):
        env_key = "DRONE_" + key.upper()
        if env_key in os.environ:
            settings[key] = float(os.environ[env_key])

    for key, value in settings.items():
        setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Final obstacle layouts
# ---------------------------------------------------------------------------

# One-obstacle curriculum obstacle used consistently for fixed, jitter, and random tasks.
ONE_OBSTACLE = Obstacle(9.0, 8.5, 2.0, 3.0)

TWO_OBSTACLES = [
    Obstacle(8.0, 7.5, 2.0, 4.0),
    Obstacle(12.5, 10.0, 2.0, 4.0),
]

THREE_OBSTACLES = [
    Obstacle(6.5, 4.5, 1.8, 5.5),
    Obstacle(12.0, 8.5, 1.8, 5.5),
    Obstacle(4.0, 14.5, 4.0, 1.2),
]


# ---------------------------------------------------------------------------
# Base task builders
# ---------------------------------------------------------------------------

def _base_20x20(
    max_steps: int = 500,
    obstacles: List[Obstacle] | None = None,
) -> EnvConfig:
    cfg = EnvConfig(
        world_size=20.0,
        dt=0.05,
        max_steps=max_steps,
        max_accel=2.0,
        max_speed=4.0,
        drone_radius=0.3,
        goal_radius=0.6,
        wind_mean=(0.0, 0.0),
        wind_std=0.0,
        n_rays=16,
        ray_max_dist=10.0,
        fixed_start=None,
        fixed_goal=None,
        min_start_goal_dist=5.0,
        jitter_start_center=None,
        jitter_goal_center=None,
        jitter_radius=0.0,
        obstacles=list(obstacles or []),
    )
    return apply_final_rewards(cfg)


def free_fixed() -> EnvConfig:
    cfg = _base_20x20(max_steps=300, obstacles=[])
    cfg.fixed_start = (2.0, 10.0)
    cfg.fixed_goal = (18.0, 10.0)
    return cfg


def free_jitter(radius: float) -> EnvConfig:
    cfg = _base_20x20(max_steps=350, obstacles=[])
    cfg.jitter_start_center = (2.0, 10.0)
    cfg.jitter_goal_center = (18.0, 10.0)
    cfg.jitter_radius = float(radius)
    return cfg


def free_random() -> EnvConfig:
    return _base_20x20(max_steps=400, obstacles=[])


def one_obstacle_fixed() -> EnvConfig:
    cfg = _base_20x20(max_steps=500, obstacles=[ONE_OBSTACLE])
    cfg.fixed_start = (3.0, 10.0)
    cfg.fixed_goal = (17.0, 10.0)
    return cfg


def one_obstacle_jitter(radius: float) -> EnvConfig:
    cfg = _base_20x20(max_steps=500, obstacles=[ONE_OBSTACLE])
    cfg.jitter_start_center = (3.0, 10.0)
    cfg.jitter_goal_center = (17.0, 10.0)
    cfg.jitter_radius = float(radius)
    return cfg


def one_obstacle_random() -> EnvConfig:
    return _base_20x20(max_steps=500, obstacles=[ONE_OBSTACLE])


def two_obstacles_random() -> EnvConfig:
    return _base_20x20(max_steps=600, obstacles=TWO_OBSTACLES)


def three_obstacles_random() -> EnvConfig:
    return _base_20x20(max_steps=600, obstacles=THREE_OBSTACLES)


def three_obstacles_wind() -> EnvConfig:
    cfg = three_obstacles_random()
    cfg.wind_mean = (0.3, 0.0)
    cfg.wind_std = 0.08
    return cfg


# ---------------------------------------------------------------------------
# Clean task registry
# ---------------------------------------------------------------------------

TASK_BUILDERS: Dict[str, Callable[[], EnvConfig]] = {
    "free_fixed": free_fixed,
    "free_jitter_r2": lambda: free_jitter(2.0),
    "free_jitter_r5": lambda: free_jitter(5.0),
    "free_jitter_r8": lambda: free_jitter(8.0),
    "free_random": free_random,
    "one_obstacle_fixed": one_obstacle_fixed,
    "one_obstacle_jitter_r2": lambda: one_obstacle_jitter(2.0),
    "one_obstacle_jitter_r4": lambda: one_obstacle_jitter(4.0),
    "one_obstacle_jitter_r6": lambda: one_obstacle_jitter(6.0),
    "one_obstacle_random": one_obstacle_random,
    "two_obstacles_random": two_obstacles_random,
    "three_obstacles_random": three_obstacles_random,
    "three_obstacles_wind": three_obstacles_wind,
}


def make_config(task: str) -> EnvConfig:
    """Create an EnvConfig from a clean final task name."""
    if task not in TASK_BUILDERS:
        raise ValueError(
            f"Unknown task '{task}'. Valid tasks are: {', '.join(task_choices())}."
        )
    return TASK_BUILDERS[task]()


def task_choices() -> List[str]:
    """All task names accepted by argparse in the final code."""
    return sorted(TASK_BUILDERS.keys())


def describe_task(task: str) -> str:
    cfg = make_config(task)
    obstacles = [(ob.x, ob.y, ob.w, ob.h) for ob in cfg.obstacles]
    return (
        f"task={task}, world={cfg.world_size}, max_steps={cfg.max_steps}, "
        f"wind_mean={cfg.wind_mean}, wind_std={cfg.wind_std}, "
        f"fixed_start={cfg.fixed_start}, fixed_goal={cfg.fixed_goal}, "
        f"jitter_radius={cfg.jitter_radius}, obstacles={obstacles}, "
        f"rewards={REWARD_SETTINGS}"
    )


def config_asdict(task: str) -> dict:
    return asdict(make_config(task))

"""
env_configs.py
Single source of truth for all final experiment tasks and reward parameters.

Clean task naming used in the final version:
    free_fixed
    free_jitter_r2
    free_jitter_r5
    free_random
    one_obstacle_fixed
    one_obstacle_jitter_r2
    one_obstacle_jitter_r4
    one_obstacle_jitter_r6
    one_obstacle_random
    two_obstacles_random
    three_obstacles_random
    three_obstacles_wind

Important design choice:
    The one-obstacle fixed/jitter/random tasks all use the SAME obstacle size
    as the previous "smallobstaclebaby" task: Obstacle(9.0, 8.5, 2.0, 3.0).
    This avoids the confusing old curriculum where the obstacle started large
    and later became smaller.
"""

from __future__ import annotations

import os 
from dataclasses import asdict
from typing import Callable, Dict, List

from core.drone_env import EnvConfig, Obstacle, easy_config, medium_config, hard_config


# ---------------------------------------------------------------------------
# Global final reward setting: used by SAC, SB3 PPO, and custom PPO.
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
    """Apply final reward values, with optional environment-variable overrides."""
    settings = dict(REWARD_SETTINGS)

    for key in list(settings.keys()):
        env_key = "DRONE_" + key.upper()
        if env_key in os.environ:
            settings[key] = float(os.environ[env_key])

    for key, value in settings.items():
        setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Obstacle layouts
# ---------------------------------------------------------------------------

# One-obstacle curriculum obstacle. This is the same size as old smallobstaclebaby.
SMALL_CENTER_OBSTACLE = Obstacle(9.0, 8.5, 2.0, 3.0)

# Optional tall obstacle kept only for ablation/backward compatibility.
TALL_CENTER_OBSTACLE = Obstacle(9.0, 7.0, 2.0, 6.0)

# Two-obstacle intermediate layout.
TWO_OBSTACLES = [
    Obstacle(8.0, 7.5, 2.0, 4.0),
    Obstacle(12.5, 10.0, 2.0, 4.0),
]

# Final target: three obstacles, old obstaclebaby/easy layout without wind.
THREE_OBSTACLES = [
    Obstacle(6.5, 4.5, 1.8, 5.5),
    Obstacle(12.0, 8.5, 1.8, 5.5),
    Obstacle(4.0, 14.5, 4.0, 1.2),
]


# ---------------------------------------------------------------------------
# Base task builders
# ---------------------------------------------------------------------------

def _base_20x20(max_steps: int = 500, obstacles: List[Obstacle] | None = None) -> EnvConfig:
    return EnvConfig(
        world_size=20.0,
        max_steps=max_steps,
        max_accel=2.0,
        max_speed=4.0,
        wind_mean=(0.0, 0.0),
        wind_std=0.0,
        n_rays=16,
        ray_max_dist=10.0,
        obstacles=list(obstacles or []),
        min_start_goal_dist=5.0,
    )


def free_fixed() -> EnvConfig:
    cfg = _base_20x20(max_steps=300, obstacles=[])
    cfg.fixed_start = (2.0, 10.0)
    cfg.fixed_goal = (18.0, 10.0)
    return apply_final_rewards(cfg)


def free_jitter(radius: float) -> EnvConfig:
    cfg = _base_20x20(max_steps=350, obstacles=[])
    cfg.jitter_start_center = (2.0, 10.0)
    cfg.jitter_goal_center = (18.0, 10.0)
    cfg.jitter_radius = float(radius)
    return apply_final_rewards(cfg)


def free_random() -> EnvConfig:
    cfg = _base_20x20(max_steps=400, obstacles=[])
    return apply_final_rewards(cfg)


def one_obstacle_fixed() -> EnvConfig:
    cfg = _base_20x20(max_steps=500, obstacles=[SMALL_CENTER_OBSTACLE])
    cfg.fixed_start = (3.0, 10.0)
    cfg.fixed_goal = (17.0, 10.0)
    return apply_final_rewards(cfg)


def one_obstacle_jitter(radius: float) -> EnvConfig:
    cfg = _base_20x20(max_steps=500, obstacles=[SMALL_CENTER_OBSTACLE])
    cfg.jitter_start_center = (3.0, 10.0)
    cfg.jitter_goal_center = (17.0, 10.0)
    cfg.jitter_radius = float(radius)
    return apply_final_rewards(cfg)


def one_obstacle_random() -> EnvConfig:
    cfg = _base_20x20(max_steps=500, obstacles=[SMALL_CENTER_OBSTACLE])
    return apply_final_rewards(cfg)


def two_obstacles_random() -> EnvConfig:
    cfg = _base_20x20(max_steps=600, obstacles=TWO_OBSTACLES)
    return apply_final_rewards(cfg)


def three_obstacles_random() -> EnvConfig:
    cfg = _base_20x20(max_steps=600, obstacles=THREE_OBSTACLES)
    return apply_final_rewards(cfg)


def three_obstacles_wind() -> EnvConfig:
    cfg = three_obstacles_random()
    cfg.wind_mean = (0.3, 0.0)
    cfg.wind_std = 0.08
    return cfg


def one_tall_obstacle_random() -> EnvConfig:
    """Optional ablation task: old singleobstaclebaby geometry."""
    cfg = _base_20x20(max_steps=500, obstacles=[TALL_CENTER_OBSTACLE])
    return apply_final_rewards(cfg)


# ---------------------------------------------------------------------------
# Registry and aliases
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
    "one_tall_obstacle_random": one_tall_obstacle_random,
    "medium": lambda: apply_final_rewards(medium_config()),
    "hard": lambda: apply_final_rewards(hard_config()),
}

# Backward-compatible aliases. Avoid these in final figures; they are here only
# so older commands/scripts do not immediately break.
ALIASES: Dict[str, str] = {
    "babyfixed": "free_fixed",
    "babyjitter2": "free_jitter_r2",
    "babyjitter5": "free_jitter_r5",
    "babyjitter": "free_jitter_r5",
    "baby": "free_random",
    "singlefixed": "one_obstacle_fixed",
    "singlejitter2": "one_obstacle_jitter_r2",
    "singlejitter4": "one_obstacle_jitter_r4",
    "singlejitter6": "one_obstacle_jitter_r6",
    "singlejitter": "one_obstacle_jitter_r2",
    "smallobstaclebaby": "one_obstacle_random",
    "smallobstaclejitter2": "one_obstacle_jitter_r2",
    "smallobstaclejitter4": "one_obstacle_jitter_r4",
    "smallobstaclejitter6": "one_obstacle_jitter_r6",
    "singleobstaclebaby": "one_tall_obstacle_random",
    "twoobstaclebaby": "two_obstacles_random",
    "obstaclebaby": "three_obstacles_random",
    "easy": "three_obstacles_wind",
}


def canonical_task_name(task: str) -> str:
    """Return the canonical clean name for a task or alias."""
    return ALIASES.get(task, task)


def make_config(task: str) -> EnvConfig:
    """Create an EnvConfig from a clean task name or backward-compatible alias."""
    name = canonical_task_name(task)
    if name not in TASK_BUILDERS:
        raise ValueError(
            f"Unknown task '{task}'. Valid clean tasks are: {', '.join(TASK_BUILDERS.keys())}. "
            f"Aliases are: {', '.join(ALIASES.keys())}."
        )
    return TASK_BUILDERS[name]()


def task_choices() -> List[str]:
    """All names accepted by argparse."""
    return sorted(list(TASK_BUILDERS.keys()) + list(ALIASES.keys()))


def describe_task(task: str) -> str:
    """Human-readable summary for logs."""
    cfg = make_config(task)
    canonical = canonical_task_name(task)
    obstacles = [(ob.x, ob.y, ob.w, ob.h) for ob in cfg.obstacles]
    return (
        f"task={canonical}, world={cfg.world_size}, max_steps={cfg.max_steps}, "
        f"wind_std={cfg.wind_std}, fixed_start={cfg.fixed_start}, fixed_goal={cfg.fixed_goal}, "
        f"jitter_radius={cfg.jitter_radius}, obstacles={obstacles}, rewards={REWARD_SETTINGS}"
    )


def config_asdict(task: str) -> dict:
    return asdict(make_config(task))

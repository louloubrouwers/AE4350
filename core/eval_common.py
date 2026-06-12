"""
eval_common.py
Shared evaluation and plotting utilities for SAC, SB3 PPO, and custom PPO.
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List

import numpy as np

from core.wrappers import unwrap_drone_env


def compute_path_length(trajectory: np.ndarray) -> float:
    if len(trajectory) > 1:
        return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    return 0.0


def run_sb3_episode(model, env, deterministic: bool = True) -> Dict:
    """Run one episode for a Stable-Baselines3 model in a VecEnv."""
    obs = env.reset()
    done = np.array([False])
    total_reward = 0.0
    last_info = {}

    drone_env = unwrap_drone_env(env)
    manual_trajectory = []
    if hasattr(drone_env, "pos"):
        manual_trajectory.append(np.asarray(drone_env.pos, dtype=np.float32).copy())

    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)
        total_reward += float(reward[0])
        last_info = infos[0]

        if "position" in last_info:
            manual_trajectory.append(np.asarray(last_info["position"], dtype=np.float32).copy())

    traj = np.asarray(manual_trajectory, dtype=np.float32)
    return {
        "return": total_reward,
        "steps": int(last_info.get("steps", len(traj))),
        "success": bool(last_info.get("reached_goal", False)),
        "collision": bool(last_info.get("collision", False)),
        "path_length": compute_path_length(traj),
        "trajectory": traj,
        "goal": np.asarray(last_info.get("goal", getattr(drone_env, "goal", np.zeros(2))), dtype=np.float32).copy(),
        "obstacles": getattr(drone_env.cfg, "obstacles", []),
        "world_size": getattr(drone_env.cfg, "world_size", 20.0),
    }


def run_custom_episode(agent, env, obs_normalizer=None, seed: int | None = None) -> Dict:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        obs_in = obs_normalizer.normalize(obs) if obs_normalizer is not None else obs
        action = agent.deterministic_action(obs_in)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

    drone_env = env.unwrapped
    traj = np.asarray(getattr(drone_env, "trajectory", []), dtype=np.float32)

    return {
        "return": total_reward,
        "steps": int(info.get("steps", len(traj))),
        "success": bool(info.get("reached_goal", False)),
        "collision": bool(info.get("collision", False)),
        "path_length": compute_path_length(traj),
        "trajectory": traj,
        "goal": np.asarray(getattr(drone_env, "goal", np.zeros(2)), dtype=np.float32).copy(),
        "obstacles": getattr(drone_env.cfg, "obstacles", []),
        "world_size": getattr(drone_env.cfg, "world_size", 20.0),
    }


def summarize_episodes(episodes: List[Dict]) -> Dict[str, float]:
    returns = [ep["return"] for ep in episodes]
    successes = [float(ep["success"]) for ep in episodes]
    collisions = [float(ep["collision"]) for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    path_lengths = [ep["path_length"] for ep in episodes]
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "steps_mean": float(np.mean(steps)),
        "path_length_mean": float(np.mean(path_lengths)),
    }


def save_evaluation_csv(episodes: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "return", "success", "collision", "steps", "path_length"],
        )
        writer.writeheader()
        for i, ep in enumerate(episodes):
            writer.writerow({
                "episode": i,
                "return": ep["return"],
                "success": int(ep["success"]),
                "collision": int(ep["collision"]),
                "steps": ep["steps"],
                "path_length": ep["path_length"],
            })


def plot_trajectories(episodes: List[Dict], out_path: str, title: str) -> None:
    if not episodes:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except Exception:
        return

    n = len(episodes)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if n == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).reshape(-1)

    for ax, ep in zip(axes, episodes):
        W = ep["world_size"]
        ax.set_xlim(0, W)
        ax.set_ylim(0, W)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for ob in ep["obstacles"]:
            ax.add_patch(
                patches.Rectangle((ob.x, ob.y), ob.w, ob.h, alpha=0.35)
            )

        traj = ep["trajectory"]
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
            ax.plot(traj[0, 0], traj[0, 1], "o", markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], "x", markersize=7)

        goal = ep["goal"]
        ax.plot(goal[0], goal[1], "*", markersize=12)

        status = "SUCCESS" if ep["success"] else "COLLISION" if ep["collision"] else "TIMEOUT"
        ax.set_title(f"{status}, steps={ep['steps']}")

    for ax in axes[len(episodes):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_eval_summary(label: str, model_path: str, task: str, n_episodes: int, metrics: Dict, csv_path: str) -> None:
    print("=" * 72)
    print(f"{label} evaluation complete")
    print("=" * 72)
    print(f"Model        : {model_path}")
    print(f"Task         : {task}")
    print(f"Episodes     : {n_episodes}")
    print(f"Success rate : {100 * metrics['success_rate']:.1f}%")
    print(f"Collision    : {100 * metrics['collision_rate']:.1f}%")
    print(f"Mean return  : {metrics['return_mean']:+.2f} +/- {metrics['return_std']:.2f}")
    print(f"Mean steps   : {metrics['steps_mean']:.1f}")
    print(f"Mean path    : {metrics['path_length_mean']:.2f}")
    print(f"CSV          : {csv_path}")
    print("=" * 72)

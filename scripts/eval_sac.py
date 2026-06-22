"""eval_sac.py -  evaluator for SB3 SAC models."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from core.env_configs import task_choices
from core.eval_common import (
    plot_trajectories,
    print_eval_summary,
    run_sb3_episode,
    save_evaluation_csv,
    summarize_episodes,
)
from scripts.train_sac import make_vec_env

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:
    raise ImportError("Install Stable-Baselines3 with: pip install stable-baselines3[extra]") from exc


def evaluate(args: argparse.Namespace) -> None:
    model_path = os.path.join(args.model_dir, f"{args.model_name}.zip")
    vecnorm_path = os.path.join(args.model_dir, "vecnormalize.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model: {model_path}")

    raw_env = make_vec_env(args.task, seed=args.seed, normalize=False, norm_reward=False)
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnormalize.pkl not found. Evaluating without normalization.")
        env = raw_env

    model = SAC.load(model_path, env=env, device="auto")
    episodes = [run_sb3_episode(model, env, deterministic=args.deterministic) for _ in range(args.episodes)]
    metrics = summarize_episodes(episodes)

    csv_path = os.path.join(args.model_dir, f"evaluation_{args.task}_{args.model_name}.csv")
    save_evaluation_csv(episodes, csv_path)

    if args.plot > 0:
        plot_path = os.path.join(args.model_dir, f"trajectories_{args.task}_{args.model_name}.png")
        plot_trajectories(episodes[: args.plot], plot_path, f"SAC trajectories: {args.task}")
        print(f"Saved trajectory plot: {plot_path}")

    print_eval_summary("SAC", model_path, args.task, args.episodes, metrics, csv_path)
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", "--difficulty", dest="task", choices=task_choices(), default="one_obstacle_fixed")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="best_model")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--plot", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

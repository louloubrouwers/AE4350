"""eval_ppo.py -  evaluator for the custom PPO baseline."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.env_configs import task_choices
from core.eval_common import (
    plot_trajectories,
    print_eval_summary,
    run_custom_episode,
    save_evaluation_csv,
    summarize_episodes,
)
from scripts.train_ppo import RunningNormalizer
from core.wrappers import make_env

CUSTOM_PPO_DIR = os.path.join(os.path.dirname(__file__), "custom_ppo")
if CUSTOM_PPO_DIR not in sys.path:
    sys.path.insert(0, CUSTOM_PPO_DIR)

from custom_ppo.custom_ppo_agent import PPOAgent  


def load_agent_and_normalizer(model_path: str, device: str = "cpu") -> tuple[PPOAgent, Optional[RunningNormalizer]]:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    agent = PPOAgent.load(model_path, device=device)
    obs_normalizer = None
    extra = payload.get("extra", {}) or {}
    if "obs_normalizer" in extra:
        obs_normalizer = RunningNormalizer((int(payload["state_dim"]),))
        obs_normalizer.load_state_dict(extra["obs_normalizer"])
    return agent, obs_normalizer


def evaluate(args: argparse.Namespace) -> None:
    model_path = os.path.join(args.model_dir, f"{args.model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model: {model_path}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent, obs_normalizer = load_agent_and_normalizer(model_path, device=device)
    env = make_env(args.task)

    episodes = [
        run_custom_episode(agent, env, obs_normalizer=obs_normalizer, seed=args.seed + ep)
        for ep in range(args.episodes)
    ]
    metrics = summarize_episodes(episodes)

    csv_path = os.path.join(args.model_dir, f"evaluation_{args.task}_{args.model_name}.csv")
    save_evaluation_csv(episodes, csv_path)

    if args.plot > 0:
        plot_path = os.path.join(args.model_dir, f"trajectories_{args.task}_{args.model_name}.png")
        plot_trajectories(episodes[: args.plot], plot_path, f"Custom PPO trajectories: {args.task}")
        print(f"Saved trajectory plot: {plot_path}")

    print_eval_summary("Custom PPO", model_path, args.task, args.episodes, metrics, csv_path)
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", "--difficulty", dest="task", choices=task_choices(), default="free_fixed")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="best_model")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--plot", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

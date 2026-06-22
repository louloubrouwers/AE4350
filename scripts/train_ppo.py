"""
train_ppo.py
self-implemented PPO baseline.

This script uses the SAME task definitions, reward settings, and wrapper logic
as SAC and SB3 PPO:

    - env_configs.py  -> tasks, obstacles, rewards
    - wrappers.py     -> make_env() and DirectionalAvoidanceWrapper

"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.env_configs import config_asdict, describe_task, task_choices
from core.wrappers import make_env
from custom_ppo.custom_ppo_agent import PPOAgent, PPOConfig
from custom_ppo.ppo_rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Default run settings
# ---------------------------------------------------------------------------

RUN_TASK = "free_fixed"
RUN_TOTAL_TIMESTEPS = 300_000
RUN_OUT_DIR = None
RUN_SEED = 0
RUN_EVAL_EPISODES = 30
RUN_EVAL_EVERY = 1


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """
    Online observation normalization with Welford-style updates.

    This normalizer is saved inside PPO checkpoints so training can be resumed
    and evaluation can use the same observation scaling.
    """

    def __init__(self, shape, eps: float = 1e-4, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = new_mean
        self.var = m_2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "clip": self.clip,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Supports both the clean normalizer format and the older M2-based format,
        so old checkpoints can still be resumed when possible.
        """
        self.mean = np.asarray(state["mean"], dtype=np.float64)

        if "var" in state:
            self.var = np.asarray(state["var"], dtype=np.float64)
            self.count = float(state.get("count", self.count))
        elif "M2" in state:
            count = int(state.get("count", 1))
            m2 = np.asarray(state["M2"], dtype=np.float64)
            self.var = m2 / max(count - 1, 1)
            self.count = float(max(count, 1))
        else:
            raise KeyError("Normalizer state must contain either 'var' or old-format 'M2'.")

        self.clip = float(state.get("clip", self.clip))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_path_length(trajectory: np.ndarray) -> float:
    if len(trajectory) > 1:
        return float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    return 0.0


@torch.no_grad()
def evaluate_policy(
    agent: PPOAgent,
    task: str,
    normalizer: Optional[RunningNormalizer],
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    env = make_env(task)

    returns = []
    successes = []
    collisions = []
    steps = []
    path_lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            obs_in = normalizer.normalize(obs) if normalizer is not None else obs
            action = agent.deterministic_action(obs_in)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        base_env = env.unwrapped
        traj = np.asarray(getattr(base_env, "trajectory", []), dtype=np.float32)

        returns.append(total_reward)
        successes.append(float(info.get("reached_goal", False)))
        collisions.append(float(info.get("collision", False)))
        steps.append(int(info.get("steps", len(traj))))
        path_lengths.append(compute_path_length(traj))

    env.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_steps_mean": float(np.mean(steps)),
        "eval_path_length_mean": float(np.mean(path_lengths)),
    }


def write_csv_header(path: str, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_learning_curve(csv_path: str, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    timesteps = [int(float(r["timesteps"])) for r in rows]
    eval_success = [100.0 * float(r["eval_success_rate"]) for r in rows]
    eval_return = [float(r["eval_return_mean"]) for r in rows]
    eval_collision = [100.0 * float(r["eval_collision_rate"]) for r in rows]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, eval_success, marker="o")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation success rate [%]")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_success.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, eval_return, marker="o")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation mean return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_return.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, eval_collision, marker="o")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation collision rate [%]")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_collision.png"), dpi=150)
    plt.close()


def load_agent_and_normalizer(path: str, device: torch.device) -> tuple[PPOAgent, Optional[RunningNormalizer]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    agent = PPOAgent.load(path, device=device)

    normalizer = None
    extra = payload.get("extra", {}) or {}
    if "obs_normalizer" in extra:
        normalizer = RunningNormalizer((int(payload["state_dim"]),))
        normalizer.load_state_dict(extra["obs_normalizer"])

    return agent, normalizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = make_env(args.task)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    if args.load_from:
        load_path = args.load_from

        if os.path.isdir(load_path):
            best_candidate = os.path.join(load_path, "best_model.pt")
            final_candidate = os.path.join(load_path, "final_model.pt")
            load_path = best_candidate if os.path.exists(best_candidate) else final_candidate

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Could not find checkpoint: {load_path}")

        agent, loaded_normalizer = load_agent_and_normalizer(load_path, device)
        obs_normalizer = loaded_normalizer or RunningNormalizer((state_dim,))

        if agent.state_dim != state_dim or agent.action_dim != action_dim:
            raise ValueError(
                f"Loaded model dimensions {agent.state_dim}/{agent.action_dim} do not match "
                f"environment dimensions {state_dim}/{action_dim}."
            )

        print(f"Loaded PPO checkpoint from: {load_path}")

    else:
        ppo_cfg = PPOConfig(
            learning_rate=args.lr,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            hidden_size=args.hidden_size,
        )
        agent = PPOAgent(state_dim, action_dim, config=ppo_cfg, device=device)
        obs_normalizer = RunningNormalizer((state_dim,))

    # When loading a checkpoint, overwrite the training hyperparameters for this run.
    agent.cfg.learning_rate = args.lr
    agent.cfg.update_epochs = args.update_epochs
    agent.cfg.minibatch_size = args.minibatch_size
    agent.cfg.max_grad_norm = args.max_grad_norm
    agent.cfg.clip_coef = args.clip_coef
    agent.cfg.value_coef = args.value_coef
    agent.cfg.entropy_coef = args.entropy_coef
    agent.cfg.target_kl = args.target_kl
    agent.cfg.gamma = args.gamma
    agent.cfg.gae_lambda = args.gae_lambda

    for group in agent.optimizer.param_groups:
        group["lr"] = args.lr

    buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=str(device),
    )

    run_dir = args.out_dir or os.path.join("experiments", "custom_ppo", f"ppo_{args.task}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "training_log.csv")
    best_path = os.path.join(run_dir, "best_model.pt")
    final_path = os.path.join(run_dir, "final_model.pt")

    fieldnames = [
        "update",
        "timesteps",
        "fps",
        "train_return_mean",
        "train_success_rate",
        "train_collision_rate",
        "train_episode_len_mean",
        "eval_return_mean",
        "eval_return_std",
        "eval_success_rate",
        "eval_collision_rate",
        "eval_steps_mean",
        "eval_path_length_mean",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "std_mean",
        "early_stop",
    ]
    write_csv_header(csv_path, fieldnames)

    print("=" * 72)
    print("CUSTOM PPO drone-navigation training")
    print("=" * 72)
    print(f"Task            : {args.task}")
    print(f"Task description: {describe_task(args.task)}")
    print(f"Device          : {device}")
    print(f"State/action dim: {state_dim}/{action_dim}")
    print(f"Rollout steps   : {args.rollout_steps}")
    print(f"Total timesteps : {args.total_timesteps}")
    print(f"Output dir      : {run_dir}")
    print(f"Load from       : {args.load_from or 'scratch'}")
    print("PPO config      :", asdict(agent.cfg))
    print("Env config      :", config_asdict(args.task))
    print("=" * 72)

    obs, info = env.reset(seed=args.seed)
    episode_done = False
    episode_return = 0.0
    episode_len = 0

    recent_returns = deque(maxlen=50)
    recent_successes = deque(maxlen=50)
    recent_collisions = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)

    total_steps = 0
    update = 0
    best_eval_success = -1.0
    best_eval_return = -1.0e12
    start_time = time.time()

    while total_steps < args.total_timesteps:
        buffer.reset()

        # Optional linear learning-rate decay.
        if args.lr_decay:
            frac = max(1.0 - total_steps / max(args.total_timesteps, 1), 0.0)
            current_lr = args.lr * frac
        else:
            current_lr = args.lr

        for group in agent.optimizer.param_groups:
            group["lr"] = current_lr

        for _ in range(args.rollout_steps):
            if episode_done:
                obs, info = env.reset()
                episode_done = False

            obs_normalizer.update(obs)
            obs_n = obs_normalizer.normalize(obs)

            action, log_prob, value = agent.select_action(obs_n)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # terminated=True is a true terminal state.
            # truncated=True is a time limit, so GAE can still bootstrap.
            buffer.add(obs_n, action, log_prob, reward, value, done=terminated)

            episode_return += float(reward)
            episode_len += 1
            total_steps += 1
            obs = next_obs

            if terminated or truncated:
                recent_returns.append(episode_return)
                recent_successes.append(float(info.get("reached_goal", False)))
                recent_collisions.append(float(info.get("collision", False)))
                recent_lengths.append(episode_len)

                episode_return = 0.0
                episode_len = 0
                episode_done = True

            if total_steps >= args.total_timesteps:
                # Finish the rollout because PPO updates expect a full buffer.
                pass

        last_done = bool(buffer.dones[buffer.rollout_steps - 1])
        last_value = agent.value(obs_normalizer.normalize(obs))
        buffer.compute_returns_and_advantages(last_value=last_value, last_done=last_done)

        stats = agent.update(buffer)
        update += 1

        if update % args.eval_every == 0 or total_steps >= args.total_timesteps:
            eval_stats = evaluate_policy(
                agent=agent,
                task=args.task,
                normalizer=obs_normalizer,
                episodes=args.eval_episodes,
                seed=args.seed + 100_000 + update * 100,
            )

            elapsed = max(time.time() - start_time, 1e-6)
            fps = total_steps / elapsed

            train_stats = {
                "train_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "train_success_rate": float(np.mean(recent_successes)) if recent_successes else 0.0,
                "train_collision_rate": float(np.mean(recent_collisions)) if recent_collisions else 0.0,
                "train_episode_len_mean": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
            }

            row = {
                "update": update,
                "timesteps": total_steps,
                "fps": round(fps, 1),
                **train_stats,
                **eval_stats,
                **stats,
            }
            append_csv_row(csv_path, fieldnames, row)

            improved = (
                eval_stats["eval_success_rate"] > best_eval_success
                or (
                    eval_stats["eval_success_rate"] == best_eval_success
                    and eval_stats["eval_return_mean"] > best_eval_return
                )
            )

            if improved:
                best_eval_success = eval_stats["eval_success_rate"]
                best_eval_return = eval_stats["eval_return_mean"]
                agent.save(
                    best_path,
                    extra={
                        "update": update,
                        "timesteps": total_steps,
                        **eval_stats,
                        "obs_normalizer": obs_normalizer.state_dict(),
                    },
                )

            print(
                f"Update {update:04d} | steps {total_steps:8d} | "
                f"train return {train_stats['train_return_mean']:+8.2f} | "
                f"train succ {train_stats['train_success_rate']*100:5.1f}% | "
                f"eval succ {eval_stats['eval_success_rate']*100:5.1f}% | "
                f"eval coll {eval_stats['eval_collision_rate']*100:5.1f}% | "
                f"eval return {eval_stats['eval_return_mean']:+8.2f} | "
                f"len {eval_stats['eval_steps_mean']:6.1f} | "
                f"std {stats.get('std_mean', float('nan')):.3f} | "
                f"kl {stats.get('approx_kl', float('nan')):.4f}"
            )

    agent.save(
        final_path,
        extra={
            "update": update,
            "timesteps": total_steps,
            "obs_normalizer": obs_normalizer.state_dict(),
        },
    )

    plot_learning_curve(csv_path, run_dir)

    env.close()

    print("=" * 72)
    print("Training complete.")
    print(f"Best model : {best_path}")
    print(f"Final model: {final_path}")
    print(f"CSV log    : {csv_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom PPO on the clean DroneNavEnv task set.")

    parser.add_argument("--task", choices=task_choices(), default=RUN_TASK)

    # Backward-friendly convenience: allows old scripts that still pass --difficulty
    # to fail less confusingly if you replace them gradually.
    parser.add_argument(
        "--difficulty",
        choices=task_choices(),
        default=None,
        help="Deprecated alias for --task. Use --task in the final code.",
    )

    parser.add_argument("--total-timesteps", type=int, default=RUN_TOTAL_TIMESTEPS)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=RUN_EVAL_EVERY, help="Evaluate every N PPO updates.")
    parser.add_argument("--eval-episodes", type=int, default=RUN_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    parser.add_argument("--out-dir", type=str, default=RUN_OUT_DIR)
    parser.add_argument("--load-from", type=str, default=None, help="Optional PPO checkpoint directory or .pt file.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--lr-decay", action="store_true", help="Use linear learning-rate decay over training.")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=15)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=128)

    args = parser.parse_args()

    if args.difficulty is not None:
        args.task = args.difficulty

    return args


if __name__ == "__main__":
    train(parse_args())

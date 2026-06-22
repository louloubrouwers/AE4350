"""
train_sac.py
Stable-Baselines3 SAC training.

Uses shared env_configs.py and wrappers.py, so SAC uses exactly the same tasks
and reward shaping as SB3 PPO and custom PPO.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.env_configs import config_asdict, describe_task, task_choices
from core.wrappers import make_env, unwrap_drone_env

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
except ImportError as exc:
    raise ImportError("Install Stable-Baselines3 with: pip install stable-baselines3[extra]") from exc


RUN_TASK = "one_obstacle_fixed"
RUN_TOTAL_TIMESTEPS = 300_000
RUN_OUT_DIR = None
RUN_SEED = 0
RUN_DEVICE = "auto"
RUN_EVAL_FREQ = 10_000
RUN_EVAL_EPISODES = 50
RUN_PROGRESS_BAR = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_vec_env(task: str, seed: int, normalize: bool = True, norm_reward: bool = True):
    def _factory():
        return make_env(task, seed=seed, monitor=True)

    env = DummyVecEnv([_factory])
    if normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
    return env


def evaluate_model(model: SAC, eval_env, n_episodes: int = 50) -> Dict[str, float]:
    returns: List[float] = []
    successes: List[float] = []
    collisions: List[float] = []
    steps: List[int] = []
    path_lengths: List[float] = []

    drone_env = unwrap_drone_env(eval_env)

    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        total_reward = 0.0
        last_info = {}
        traj = []
        if hasattr(drone_env, "pos"):
            traj.append(np.asarray(drone_env.pos, dtype=np.float32).copy())

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            total_reward += float(reward[0])
            last_info = infos[0]
            if "position" in last_info:
                traj.append(np.asarray(last_info["position"], dtype=np.float32).copy())

        traj_arr = np.asarray(traj, dtype=np.float32)
        path_len = float(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0

        returns.append(total_reward)
        successes.append(float(last_info.get("reached_goal", False)))
        collisions.append(float(last_info.get("collision", False)))
        steps.append(int(last_info.get("steps", len(traj_arr))))
        path_lengths.append(path_len)

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_steps_mean": float(np.mean(steps)),
        "eval_path_length_mean": float(np.mean(path_lengths)),
    }


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, out_dir: str, eval_freq: int, n_eval_episodes: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = os.path.join(out_dir, "training_log.csv")
        self.best_success = -1.0
        self.best_return = -1e12
        self.start_time = time.time()
        self.fieldnames = [
            "timesteps", "fps", "eval_return_mean", "eval_return_std",
            "eval_success_rate", "eval_collision_rate", "eval_steps_mean", "eval_path_length_mean",
        ]

    def _init_callback(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass

        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.training = False
            self.eval_env.norm_reward = False

        metrics = evaluate_model(self.model, self.eval_env, self.n_eval_episodes)
        fps = int(self.num_timesteps / max(time.time() - self.start_time, 1e-8))
        row = {"timesteps": self.num_timesteps, "fps": fps, **metrics}

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

        improved = (
            metrics["eval_success_rate"] > self.best_success or
            (metrics["eval_success_rate"] == self.best_success and metrics["eval_return_mean"] > self.best_return)
        )
        if improved:
            self.best_success = metrics["eval_success_rate"]
            self.best_return = metrics["eval_return_mean"]
            self.model.save(os.path.join(self.out_dir, "best_model"))
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.out_dir, "vecnormalize.pkl"))

        if self.verbose:
            print(
                f"Eval @ {self.num_timesteps:>7d} | "
                f"succ {metrics['eval_success_rate'] * 100:5.1f}% | "
                f"ret {metrics['eval_return_mean']:+8.2f} | "
                f"coll {metrics['eval_collision_rate'] * 100:5.1f}% | "
                f"len {metrics['eval_steps_mean']:6.1f} | path {metrics['eval_path_length_mean']:6.2f}"
            )
        return True


def plot_learning_curve(csv_path: str, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    x = [int(float(r["timesteps"])) for r in rows]
    y_success = [float(r["eval_success_rate"]) for r in rows]
    y_return = [float(r["eval_return_mean"]) for r in rows]
    plt.figure(figsize=(8, 4.5)); plt.plot(x, y_success, marker="o"); plt.xlabel("steps"); plt.ylabel("success rate"); plt.ylim(-0.05, 1.05); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "learning_curve_success.png"), dpi=150); plt.close()
    plt.figure(figsize=(8, 4.5)); plt.plot(x, y_return, marker="o"); plt.xlabel("steps"); plt.ylabel("mean return"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "learning_curve_return.png"), dpi=150); plt.close()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_env = make_vec_env(args.task, args.seed, normalize=True, norm_reward=True)
    eval_env = make_vec_env(args.task, args.seed + 10_000, normalize=True, norm_reward=False)

    if args.load_from:
        stats_path = os.path.join(args.load_from, "vecnormalize.pkl")
        if os.path.exists(stats_path):
            print(f"Loading VecNormalize stats from: {stats_path}")
            train_env = VecNormalize.load(stats_path, DummyVecEnv([lambda: make_env(args.task, seed=args.seed, monitor=True)]))
            train_env.training = True
            train_env.norm_reward = True
            eval_env = VecNormalize.load(stats_path, DummyVecEnv([lambda: make_env(args.task, seed=args.seed + 10_000, monitor=True)]))
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            print(f"WARNING: no VecNormalize stats found at {stats_path}")

    policy_kwargs = dict(net_arch=dict(pi=[args.hidden_size, args.hidden_size], qf=[args.hidden_size, args.hidden_size]))

    if args.load_from:
        load_path = os.path.join(args.load_from, "best_model.zip")
        if not os.path.exists(load_path):
            load_path = os.path.join(args.load_from, "final_model.zip")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No best_model.zip or final_model.zip found in {args.load_from}")
        print(f"Loading previous SAC model from: {load_path}")
        model = SAC.load(
            load_path,
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            device=args.device,
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    with open(os.path.join(args.out_dir, "run_config.txt"), "w") as f:
        f.write("SB3 SAC clean training\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + describe_task(args.task) + "\n")
        f.write(str(config_asdict(args.task)) + "\n")

    print("=" * 72)
    print("SB3 SAC clean training")
    print(f"Task            : {args.task}")
    print(f"Total timesteps : {args.total_timesteps}")
    print(f"Output dir      : {args.out_dir}")
    print("=" * 72)

    callback = EvalCallback(eval_env, args.out_dir, args.eval_freq, args.eval_episodes)
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=args.progress_bar)
    model.save(os.path.join(args.out_dir, "final_model"))
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(args.out_dir, "vecnormalize.pkl"))
    plot_learning_curve(os.path.join(args.out_dir, "training_log.csv"), args.out_dir)
    train_env.close(); eval_env.close()
    print("Training complete.")


def parse_args() -> argparse.Namespace:
    if len(__import__("sys").argv) == 1:
        out_dir = RUN_OUT_DIR or os.path.join("experiments", "sac", RUN_TASK)
        return argparse.Namespace(
            task=RUN_TASK, total_timesteps=RUN_TOTAL_TIMESTEPS, out_dir=out_dir,
            seed=RUN_SEED, device=RUN_DEVICE, learning_rate=3e-4, buffer_size=300_000,
            learning_starts=5_000, batch_size=256, tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1, ent_coef="auto", hidden_size=128,
            eval_freq=RUN_EVAL_FREQ, eval_episodes=RUN_EVAL_EPISODES,
            progress_bar=RUN_PROGRESS_BAR, load_from=None,
        )
    p = argparse.ArgumentParser()
    p.add_argument("--task", "--difficulty", dest="task", choices=task_choices(), default="one_obstacle_fixed")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--ent-coef", type=str, default="auto")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--load-from", type=str, default=None)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("experiments", "sac", args.task)
    return args


if __name__ == "__main__":
    train(parse_args())

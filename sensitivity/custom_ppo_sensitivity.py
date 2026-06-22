"""
Runs the final sensitivity analysis for the self-implemented PPO controller.

Purpose:
    Test how sensitive custom PPO is when adapting from a free-random navigation
    policy to a randomized one-obstacle avoidance task.

Final baseline:
    lr       = 3e-4
    entropy  = 0.003
    clip     = 0.2
    reward   = r_goal=350, r_progress=4, r_collision=-100,
               r_proximity=-0.01, proximity_threshold=1.0

Sensitivity values:
    learning rate: 0.00015 / 0.0003 / 0.001
    entropy:      0.001 / 0.003 / 0.008
    PPO clip:     0.05 / 0.2 / 0.4

Reward cases:
    baseline
    weak guidance:       lower goal/progress shaping
    aggressive goal:     stronger goal/progress shaping
    weak safety:         weaker collision/proximity penalties
    safety focused:      stronger collision/proximity penalties

"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from core import env_configs 

REWARD_KEYS = [
    "r_goal",
    "r_collision",
    "r_step",
    "r_action",
    "r_progress",
    "r_proximity",
    "proximity_threshold",
]


def build_cases() -> List[Dict]:
    """Sensitivity cases used for the custom PPO runs."""
    base = {
        "lr": 3e-4,
        "entropy": 0.003,
        "clip": 0.2,
        "r_goal": 350.0,
        "r_collision": -100.0,
        "r_step": -0.01,
        "r_action": -0.001,
        "r_progress": 4.0,
        "r_proximity": -0.01,
        "proximity_threshold": 1.0,
    }

    cases = [
        {
            "name": "baseline_lr3e4_ent003_clip02_reward350_4",
            "group": "baseline",
            **base,
        },

        # Learning-rate sensitivity
        {
            "name": "lr_low_1p5e4",
            "group": "learning_rate",
            **{**base, "lr": 1.5e-4},
        },
        {
            "name": "lr_high_1e3",
            "group": "learning_rate",
            **{**base, "lr": 1e-3},
        },

        # Entropy sensitivity
        {
            "name": "entropy_low_001",
            "group": "entropy",
            **{**base, "entropy": 0.001},
        },
        {
            "name": "entropy_high_008",
            "group": "entropy",
            **{**base, "entropy": 0.008},
        },

        # PPO clipping sensitivity
        {
            "name": "clip_very_low_005",
            "group": "ppo_clip",
            **{**base, "clip": 0.05},
        },
        {
            "name": "clip_very_high_04",
            "group": "ppo_clip",
            **{**base, "clip": 0.40},
        },

        # Reward-design sensitivity
        {
            "name": "reward_weak_guidance",
            "group": "reward",
            **{
                **base,
                "r_goal": 200.0,
                "r_progress": 1.0,
                "r_collision": -100.0,
                "r_proximity": -0.01,
                "proximity_threshold": 1.0,
            },
        },
        {
            "name": "reward_aggressive_goal",
            "group": "reward",
            **{
                **base,
                "r_goal": 600.0,
                "r_progress": 10.0,
                "r_collision": -100.0,
                "r_proximity": -0.01,
                "proximity_threshold": 1.0,
            },
        },
        {
            "name": "reward_weak_safety",
            "group": "reward",
            **{
                **base,
                "r_goal": 350.0,
                "r_progress": 4.0,
                "r_collision": -50.0,
                "r_proximity": 0.0,
                "proximity_threshold": 1.0,
            },
        },
        {
            "name": "reward_safety_focused",
            "group": "reward",
            **{
                **base,
                "r_goal": 350.0,
                "r_progress": 4.0,
                "r_collision": -250.0,
                "r_proximity": -0.05,
                "proximity_threshold": 1.5,
            },
        },
    ]

    return cases


def set_reward_env(env: Dict[str, str], case: Dict) -> None:
    for key in REWARD_KEYS:
        env["DRONE_" + key.upper()] = str(case[key])


def set_baseline_reward_env(env: Dict[str, str]) -> None:
    env["DRONE_R_GOAL"] = "350.0"
    env["DRONE_R_COLLISION"] = "-100.0"
    env["DRONE_R_STEP"] = "-0.01"
    env["DRONE_R_ACTION"] = "-0.001"
    env["DRONE_R_PROGRESS"] = "4.0"
    env["DRONE_R_PROXIMITY"] = "-0.01"
    env["DRONE_PROXIMITY_THRESHOLD"] = "1.0"


def run_command(cmd: List[str], env: Optional[Dict[str, str]] = None, dry_run: bool = False) -> int:
    print("\n" + "=" * 96)
    print(" ".join(cmd))
    print("=" * 96)

    if dry_run:
        return 0

    completed = subprocess.run(cmd, env=env)
    return int(completed.returncode)


def check_reward_override(task: str) -> None:
    """Check that reward overrides are picked up by the task config."""
   

    old_env = {key: os.environ.get("DRONE_" + key.upper()) for key in REWARD_KEYS}

    try:
        test_values = {
            "r_goal": 123.0,
            "r_collision": -321.0,
            "r_step": -0.123,
            "r_action": -0.456,
            "r_progress": 7.0,
            "r_proximity": -0.789,
            "proximity_threshold": 1.75,
        }

        for key, value in test_values.items():
            os.environ["DRONE_" + key.upper()] = str(value)

        cfg = env_configs.make_config(task)

        for key, expected in test_values.items():
            actual = float(getattr(cfg, key))
            if abs(actual - expected) > 1e-9:
                raise RuntimeError(
                    f"Reward override check failed for {key}: expected {expected}, got {actual}. "
                    "Check apply_final_rewards in env_configs.py."
                )

        print("Reward override check passed for all reward keys.")

    finally:
        for key, old_value in old_env.items():
            env_key = "DRONE_" + key.upper()
            if old_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_value


def write_case_summary(base_out: Path, cases: List[Dict], args: argparse.Namespace) -> None:
    base_out.mkdir(parents=True, exist_ok=True)
    path = base_out / "sensitivity_cases.csv"

    fieldnames = [
        "name",
        "group",
        "task",
        "parent_dir",
        "timesteps",
        "seed",
        "lr",
        "entropy",
        "clip",
        "r_goal",
        "r_collision",
        "r_step",
        "r_action",
        "r_progress",
        "r_proximity",
        "proximity_threshold",
        "out_dir",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            writer.writerow(
                {
                    "name": case["name"],
                    "group": case["group"],
                    "task": args.task,
                    "parent_dir": args.parent_dir,
                    "timesteps": args.timesteps,
                    "seed": args.seed,
                    "lr": case["lr"],
                    "entropy": case["entropy"],
                    "clip": case["clip"],
                    "r_goal": case["r_goal"],
                    "r_collision": case["r_collision"],
                    "r_step": case["r_step"],
                    "r_action": case["r_action"],
                    "r_progress": case["r_progress"],
                    "r_proximity": case["r_proximity"],
                    "proximity_threshold": case["proximity_threshold"],
                    "out_dir": str(base_out / case["name"]),
                }
            )

    print(f"Wrote case summary: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run final custom PPO sensitivity analysis.")

    p.add_argument("--parent-dir", type=str, required=True, help="Checkpoint directory to fine-tune from.")
    p.add_argument("--task", type=str, default="one_obstacle_random")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--base-out", type=str, default=os.path.join("experiments", "sensitivity_clean"))
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--eval-seed", type=int, default=900)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--plot", type=int, default=20)
    p.add_argument("--train-script", type=str, default="scripts/train_ppo.py")
    p.add_argument("--eval-script", type=str, default="scripts/eval_ppo.py")
    p.add_argument("--model-name", type=str, default="final_model")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--eval-with-own-reward",
        action="store_true",
        help=(
            "If set, reward-trained variants are evaluated with their own reward settings. "
            "Default is false: all policies are evaluated under baseline reward settings "
            "so returns are comparable."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_out = Path(args.base_out)
    cases = build_cases()
    write_case_summary(base_out, cases, args)

    if not args.dry_run:
        check_reward_override(args.task)

    python_exe = sys.executable

    for case in cases:
        out_dir = base_out / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        train_env = os.environ.copy()
        set_reward_env(train_env, case)

        train_cmd = [
            python_exe,
            args.train_script,
            "--task", args.task,
            "--load-from", args.parent_dir,
            "--total-timesteps", str(args.timesteps),
            "--out-dir", str(out_dir),
            "--seed", str(args.seed),
            "--lr", str(case["lr"]),
            "--entropy-coef", str(case["entropy"]),
            "--clip-coef", str(case["clip"]),
            "--eval-episodes", "30",
        ]

        if args.cpu:
            train_cmd.append("--cpu")

        if not args.skip_train:
            ret = run_command(train_cmd, env=train_env, dry_run=args.dry_run)
            if ret != 0:
                raise RuntimeError(f"Training failed for case {case['name']} with return code {ret}")

        eval_env = os.environ.copy()

        if args.eval_with_own_reward:
            set_reward_env(eval_env, case)
        else:
            set_baseline_reward_env(eval_env)

        eval_cmd = [
            python_exe,
            args.eval_script,
            "--task", args.task,
            "--model-dir", str(out_dir),
            "--model-name", args.model_name,
            "--episodes", str(args.eval_episodes),
            "--plot", str(args.plot),
            "--seed", str(args.eval_seed),
        ]

        if not args.skip_eval:
            ret = run_command(eval_cmd, env=eval_env, dry_run=args.dry_run)
            if ret != 0:
                raise RuntimeError(f"Evaluation failed for case {case['name']} with return code {ret}")

    print("\nAll final sensitivity cases completed.")
    print(f"Results folder: {base_out}")
    print("Use sensitivity_cases.csv and the generated evaluation_*.csv files for the report table.")


if __name__ == "__main__":
    main()

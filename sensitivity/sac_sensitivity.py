"""
sac_sensitivity.py

Corrected SAC sensitivity script:
- Removes manual entropy-coefficient cases, because the parent SAC model was trained
  with ent_coef="auto". Continuing from an auto-entropy SAC checkpoint with fixed
  ent_coef values can break Stable-Baselines3 loading.
- Keeps only learning-rate and reward-design sensitivity.
- Skips cases that already have an evaluation CSV, so completed baseline / LR
  runs are not repeated.
- If a case already has a trained final_model but no evaluation CSV, it only runs
  evaluation for that case.

Recommended command:
    py sac_sensitivity.py --parent-dir experiments/sac/two_obstacles_random --task three_obstacles_random --timesteps 500000 --base-out experiments/sac_sensitivity_clean
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


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
    """
    Final SAC sensitivity matrix.

    Entropy sensitivity is intentionally not included. SAC uses automatic entropy
    tuning in the baseline and parent checkpoint. Fixed entropy coefficients are
    not directly comparable when continuing from that parent model.
    """
    base = {
        "learning_rate": 3e-4,
        "ent_coef": "auto",
        "r_goal": 350.0,
        "r_collision": -100.0,
        "r_step": -0.01,
        "r_action": -0.001,
        "r_progress": 4.0,
        "r_proximity": -0.01,
        "proximity_threshold": 1.0,
    }

    cases = [
        {"name": "baseline_lr3e4_entauto_reward350_4", "group": "baseline", **base},

        {"name": "lr_low_1e4", "group": "learning_rate", **{**base, "learning_rate": 1e-4}},
        {"name": "lr_high_1e3", "group": "learning_rate", **{**base, "learning_rate": 1e-3}},

        {
            "name": "reward_weak_guidance",
            "group": "reward",
            **{**base, "r_goal": 200.0, "r_progress": 1.0},
        },
        {
            "name": "reward_aggressive_goal",
            "group": "reward",
            **{**base, "r_goal": 600.0, "r_progress": 10.0},
        },
        {
            "name": "reward_weak_safety",
            "group": "reward",
            **{**base, "r_collision": -50.0, "r_proximity": 0.0},
        },
        {
            "name": "reward_safety_focused",
            "group": "reward",
            **{**base, "r_collision": -250.0, "r_proximity": -0.05, "proximity_threshold": 1.5},
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
    """
    Verifies that env_configs.py reads all DRONE_* reward overrides.
    Works both before and after moving env_configs.py into core/.
    """
    try:
        import core.env_configs
    except Exception:
        try:
            from core import env_configs
        except Exception as exc:
            print(f"WARNING: could not import env_configs.py for reward-override check: {exc}")
            return

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
    path = base_out / "sac_sensitivity_cases.csv"

    fieldnames = [
        "name", "group", "task", "parent_dir", "timesteps", "seed",
        "learning_rate", "ent_coef",
        "r_goal", "r_collision", "r_step", "r_action", "r_progress",
        "r_proximity", "proximity_threshold", "out_dir",
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
                    "learning_rate": case["learning_rate"],
                    "ent_coef": case["ent_coef"],
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
    p = argparse.ArgumentParser(description="Run corrected SAC sensitivity analysis.")

    p.add_argument("--parent-dir", type=str, required=True, help="Checkpoint directory to fine-tune from.")
    p.add_argument("--task", type=str, default="three_obstacles_random")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--base-out", type=str, default=os.path.join("experiments", "sac_sensitivity_clean"))
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--eval-seed", type=int, default=950)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--plot", type=int, default=20)
    p.add_argument("--train-script", type=str, default="scripts/train_sac.py")
    p.add_argument("--eval-script", type=str, default="scripts/eval_sac.py")
    p.add_argument("--model-name", type=str, default="final_model")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--eval-with-own-reward",
        action="store_true",
        help="Evaluate reward variants under their own reward settings. Default: evaluate all under baseline rewards.",
    )

    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--train-eval-episodes", type=int, default=50)

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

        eval_csv = out_dir / f"evaluation_{args.task}_{args.model_name}.csv"
        model_zip = out_dir / f"{args.model_name}.zip"

        if eval_csv.exists():
            print(f"\nSkipping {case['name']} because evaluation already exists:")
            print(f"  {eval_csv}")
            continue

        if model_zip.exists():
            print(f"\nSkipping training for {case['name']} because model already exists:")
            print(f"  {model_zip}")
        else:
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
                "--device", args.device,
                "--learning-rate", str(case["learning_rate"]),
                "--ent-coef", str(case["ent_coef"]),
                "--buffer-size", str(args.buffer_size),
                "--learning-starts", str(args.learning_starts),
                "--batch-size", str(args.batch_size),
                "--tau", str(args.tau),
                "--gamma", str(args.gamma),
                "--train-freq", str(args.train_freq),
                "--gradient-steps", str(args.gradient_steps),
                "--hidden-size", str(args.hidden_size),
                "--eval-freq", str(args.eval_freq),
                "--eval-episodes", str(args.train_eval_episodes),
            ]

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

    print("\nAll corrected SAC sensitivity cases completed.")
    print(f"Results folder: {base_out}")
    print("Use sac_sensitivity_cases.csv and the generated evaluation_*.csv files for the report table.")


if __name__ == "__main__":
    main()

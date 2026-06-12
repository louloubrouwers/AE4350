"""
evaluate_all_existing_models.py

Evaluate all existing SAC, SB3 PPO, and custom PPO model folders and create one
summary CSV table with success rate, collision rate, mean return, mean steps, and
mean path length.

Default behavior:
- Scans:
    experiments/sac
    experiments/sb3_ppo
    experiments/custom_ppo
- For each model folder, infers the task from the folder name using the clean
  task names from core.env_configs.
- Evaluates best_model if it exists, otherwise final_model.
- Skips evaluations whose CSV already exists.
- Writes:
    experiments/all_model_evaluations_summary.csv

Run from the project root:
    py scripts/evaluate_all_existing_models.py --episodes 200 --plot 20 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.env_configs import task_choices


CLEAN_TASKS = [
    "three_obstacles_wind",
    "three_obstacles_random",
    "two_obstacles_random",
    "one_obstacle_jitter_r6",
    "one_obstacle_jitter_r4",
    "one_obstacle_jitter_r2",
    "one_obstacle_random",
    "one_obstacle_fixed",
    "free_jitter_r8",
    "free_jitter_r5",
    "free_jitter_r2",
    "free_random",
    "free_fixed",
]


ALGORITHMS = {
    "SAC": {
        "base_dir": ROOT / "experiments" / "sac",
        "eval_script": ROOT / "scripts" / "eval_sac.py",
        "extension": ".zip",
    },
    "SB3 PPO": {
        "base_dir": ROOT / "experiments" / "sb3_ppo",
        "eval_script": ROOT / "scripts" / "eval_sb3_ppo.py",
        "extension": ".zip",
    },
    "Custom PPO": {
        "base_dir": ROOT / "experiments" / "custom_ppo",
        "eval_script": ROOT / "scripts" / "eval_ppo.py",
        "extension": ".pt",
    },
}


def infer_task_from_path(path: Path) -> Optional[str]:
    """
    Infer the task from the model directory name.

    Uses longest-match first, so:
        one_obstacle_random_from_free_random
    becomes:
        one_obstacle_random
    not:
        free_random
    """
    text = path.as_posix().lower()
    for task in sorted(CLEAN_TASKS, key=len, reverse=True):
        if task.lower() in text:
            return task
    return None


def find_model_name(model_dir: Path, extension: str, preferred: str) -> Optional[str]:
    """
    Choose model name without extension.
    Preferred is usually best_model, then fallback to final_model.
    """
    candidates = [preferred, "best_model", "final_model"]
    seen = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (model_dir / f"{candidate}{extension}").exists():
            return candidate

    return None


def has_any_model_file(model_dir: Path, extension: str) -> bool:
    return any(model_dir.glob(f"*{extension}"))


def discover_model_dirs(base_dir: Path, extension: str) -> List[Path]:
    """
    Return all directories under base_dir that contain at least one model file.
    """
    if not base_dir.exists():
        return []

    dirs = []
    for path in base_dir.rglob("*"):
        if path.is_dir() and has_any_model_file(path, extension):
            dirs.append(path)

    # Also include base_dir itself if it directly contains a model.
    if has_any_model_file(base_dir, extension):
        dirs.append(base_dir)

    return sorted(set(dirs), key=lambda p: p.as_posix().lower())


def evaluation_csv_path(model_dir: Path, task: str, model_name: str) -> Path:
    return model_dir / f"evaluation_{task}_{model_name}.csv"


def read_metrics(csv_path: Path) -> Optional[Dict[str, float]]:
    if not csv_path.exists():
        return None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    returns = []
    successes = []
    collisions = []
    steps = []
    path_lengths = []

    for row in rows:
        if "return" in row and row["return"] != "":
            returns.append(float(row["return"]))
        if "success" in row and row["success"] != "":
            successes.append(float(row["success"]))
        if "collision" in row and row["collision"] != "":
            collisions.append(float(row["collision"]))
        if "steps" in row and row["steps"] != "":
            steps.append(float(row["steps"]))
        if "path_length" in row and row["path_length"] != "":
            path_lengths.append(float(row["path_length"]))

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "episodes": len(rows),
        "success_rate": mean(successes),
        "collision_rate": mean(collisions),
        "return_mean": mean(returns),
        "steps_mean": mean(steps),
        "path_length_mean": mean(path_lengths),
    }


def run_eval(
    algorithm: str,
    eval_script: Path,
    task: str,
    model_dir: Path,
    model_name: str,
    episodes: int,
    plot: int,
    seed: int,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(eval_script),
        "--task",
        task,
        "--model-dir",
        str(model_dir),
        "--model-name",
        model_name,
        "--episodes",
        str(episodes),
        "--plot",
        str(plot),
        "--seed",
        str(seed),
    ]

    print("\n" + "=" * 100)
    print(f"{algorithm} | task={task} | model_dir={model_dir} | model={model_name}")
    print(" ".join(cmd))
    print("=" * 100)

    if dry_run:
        return 0

    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def write_summary(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "task",
        "model_name",
        "model_dir",
        "episodes",
        "success_rate_percent",
        "collision_rate_percent",
        "mean_return",
        "mean_steps",
        "mean_path_length",
        "evaluation_csv",
        "status",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n" + "=" * 100)
    print(f"Wrote summary table: {out_csv}")
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--plot", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--preferred-model", type=str, default="best_model", choices=["best_model", "final_model"])
    p.add_argument("--out-csv", type=str, default=str(ROOT / "experiments" / "all_model_evaluations_summary.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true", help="Rerun evaluations even if evaluation CSV already exists.")

    p.add_argument(
        "--algorithms",
        nargs="*",
        default=["SAC", "SB3 PPO", "Custom PPO"],
        choices=["SAC", "SB3 PPO", "Custom PPO"],
        help="Algorithms to scan.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = []

    valid_tasks = set(task_choices())

    for algorithm in args.algorithms:
        info = ALGORITHMS[algorithm]
        base_dir = info["base_dir"]
        eval_script = info["eval_script"]
        extension = info["extension"]

        model_dirs = discover_model_dirs(base_dir, extension)

        print(f"\nFound {len(model_dirs)} {algorithm} model folders under {base_dir}")

        for model_dir in model_dirs:
            task = infer_task_from_path(model_dir)
            if task is None:
                print(f"Skipping because task could not be inferred: {model_dir}")
                continue

            if task not in valid_tasks:
                print(f"Skipping invalid task '{task}' inferred from {model_dir}")
                continue

            model_name = find_model_name(model_dir, extension, args.preferred_model)
            if model_name is None:
                print(f"Skipping because no best_model/final_model found: {model_dir}")
                continue

            eval_csv = evaluation_csv_path(model_dir, task, model_name)

            status = "existing"
            if args.rerun or not eval_csv.exists():
                ret = run_eval(
                    algorithm=algorithm,
                    eval_script=eval_script,
                    task=task,
                    model_dir=model_dir,
                    model_name=model_name,
                    episodes=args.episodes,
                    plot=args.plot,
                    seed=args.seed,
                    dry_run=args.dry_run,
                )
                if ret != 0:
                    rows.append({
                        "algorithm": algorithm,
                        "task": task,
                        "model_name": model_name,
                        "model_dir": str(model_dir.relative_to(ROOT)),
                        "episodes": "",
                        "success_rate_percent": "",
                        "collision_rate_percent": "",
                        "mean_return": "",
                        "mean_steps": "",
                        "mean_path_length": "",
                        "evaluation_csv": str(eval_csv.relative_to(ROOT)),
                        "status": f"evaluation_failed_return_code_{ret}",
                    })
                    continue
                status = "evaluated"

            metrics = None if args.dry_run else read_metrics(eval_csv)

            if metrics is None:
                rows.append({
                    "algorithm": algorithm,
                    "task": task,
                    "model_name": model_name,
                    "model_dir": str(model_dir.relative_to(ROOT)),
                    "episodes": "",
                    "success_rate_percent": "",
                    "collision_rate_percent": "",
                    "mean_return": "",
                    "mean_steps": "",
                    "mean_path_length": "",
                    "evaluation_csv": str(eval_csv.relative_to(ROOT)),
                    "status": "missing_or_unreadable_csv" if not args.dry_run else "dry_run",
                })
            else:
                rows.append({
                    "algorithm": algorithm,
                    "task": task,
                    "model_name": model_name,
                    "model_dir": str(model_dir.relative_to(ROOT)),
                    "episodes": int(metrics["episodes"]),
                    "success_rate_percent": round(100.0 * metrics["success_rate"], 2),
                    "collision_rate_percent": round(100.0 * metrics["collision_rate"], 2),
                    "mean_return": round(metrics["return_mean"], 3),
                    "mean_steps": round(metrics["steps_mean"], 3),
                    "mean_path_length": round(metrics["path_length_mean"], 3),
                    "evaluation_csv": str(eval_csv.relative_to(ROOT)),
                    "status": status,
                })

    out_csv = Path(args.out_csv)
    write_summary(rows, out_csv)

    print("\nQuick table:")
    for row in rows:
        print(
            f"{row['algorithm']:10s} | {row['task']:24s} | "
            f"{row['model_name']:11s} | success={row['success_rate_percent']} | "
            f"collision={row['collision_rate_percent']} | status={row['status']}"
        )


if __name__ == "__main__":
    main()

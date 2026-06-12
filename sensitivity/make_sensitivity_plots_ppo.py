"""
make_sensitivity_plots_corrected.py

Creates summary tables and comparative plots for the corrected custom PPO
sensitivity run.

Expected folder:
    experiments/sensitivity/custom_ppo_one_obstacle_random_corrected/

Outputs:
    sensitivity_summary.csv
    comparative_plots/success_learning_rate.png
    comparative_plots/collision_learning_rate.png
    comparative_plots/return_learning_rate.png
    comparative_plots/success_entropy.png
    comparative_plots/success_ppo_clip.png
    comparative_plots/success_reward.png
    comparative_plots/final_success_by_case.png
    comparative_plots/final_collision_by_case.png

Example:
    py make_sensitivity_plots_corrected.py ^
        --base-dir experiments/sensitivity/custom_ppo_one_obstacle_random_corrected ^
        --task one_obstacle_random ^
        --model-name final_model
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def nice_label(row: Dict[str, str]) -> str:
    group = row["group"]

    if group == "baseline":
        return "baseline"

    if group == "learning_rate":
        return f"lr={float(row['lr']):g}"

    if group == "entropy":
        return f"entropy={float(row['entropy']):g}"

    if group == "ppo_clip":
        return f"clip={float(row['clip']):g}"

    if group == "reward":
        name = row["name"]
        if name == "reward_weak_guidance":
            return "weak guidance"
        if name == "reward_aggressive_goal":
            return "aggressive goal"
        if name == "reward_weak_safety":
            return "weak safety"
        if name == "reward_safety_focused":
            return "safety focused"
        return f"reward={float(row['r_goal']):g}/{float(row['r_progress']):g}"

    return row["name"]


def case_sort_key(row: Dict[str, str]):
    group_order = {
        "baseline": 0,
        "learning_rate": 1,
        "entropy": 2,
        "ppo_clip": 3,
        "reward": 4,
    }
    group = row["group"]

    if group == "learning_rate":
        return (group_order[group], float(row["lr"]))

    if group == "entropy":
        return (group_order[group], float(row["entropy"]))

    if group == "ppo_clip":
        return (group_order[group], float(row["clip"]))

    if group == "reward":
        reward_order = {
            "reward_weak_guidance": 1,
            "reward_aggressive_goal": 2,
            "reward_weak_safety": 3,
            "reward_safety_focused": 4,
        }
        return (group_order[group], reward_order.get(row["name"], 99))

    return (group_order.get(group, 99), row["name"])


def read_final_eval(case_dir: Path, task: str, model_name: str) -> Dict[str, float]:
    eval_path = case_dir / f"evaluation_{task}_{model_name}.csv"
    rows = read_csv(eval_path)

    if not rows:
        return {
            "episodes": 0,
            "success_rate": np.nan,
            "collision_rate": np.nan,
            "mean_return": np.nan,
            "std_return": np.nan,
            "mean_steps": np.nan,
            "mean_path_length": np.nan,
        }

    returns = np.array([to_float(r.get("return", "")) for r in rows], dtype=float)
    successes = np.array([to_float(r.get("success", "")) for r in rows], dtype=float)
    collisions = np.array([to_float(r.get("collision", "")) for r in rows], dtype=float)
    steps = np.array([to_float(r.get("steps", "")) for r in rows], dtype=float)
    path_lengths = np.array([to_float(r.get("path_length", "")) for r in rows], dtype=float)

    return {
        "episodes": len(rows),
        "success_rate": float(np.nanmean(successes)),
        "collision_rate": float(np.nanmean(collisions)),
        "mean_return": float(np.nanmean(returns)),
        "std_return": float(np.nanstd(returns)),
        "mean_steps": float(np.nanmean(steps)),
        "mean_path_length": float(np.nanmean(path_lengths)),
    }


def load_training_curve(case_dir: Path) -> Dict[str, np.ndarray]:
    rows = read_csv(case_dir / "training_log.csv")

    if not rows:
        return {
            "timesteps": np.array([]),
            "eval_success_rate": np.array([]),
            "eval_collision_rate": np.array([]),
            "eval_return_mean": np.array([]),
        }

    return {
        "timesteps": np.array([to_float(r.get("timesteps", "")) for r in rows], dtype=float),
        "eval_success_rate": np.array([to_float(r.get("eval_success_rate", "")) for r in rows], dtype=float),
        "eval_collision_rate": np.array([to_float(r.get("eval_collision_rate", "")) for r in rows], dtype=float),
        "eval_return_mean": np.array([to_float(r.get("eval_return_mean", "")) for r in rows], dtype=float),
    }


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_group_curves(
    rows: List[Dict[str, str]],
    base_dir: Path,
    plots_dir: Path,
    group: str,
    metric: str,
    ylabel: str,
    filename: str,
    percent: bool = False,
    smooth_window: int = 1,
) -> None:
    group_rows = [r for r in rows if r["group"] == group or r["group"] == "baseline"]
    group_rows = sorted(group_rows, key=case_sort_key)

    plt.figure(figsize=(8, 4.8))

    plotted = False
    for row in group_rows:
        case_dir = base_dir / row["name"]
        curve = load_training_curve(case_dir)
        x = curve["timesteps"]
        y = curve[metric]

        if len(x) == 0:
            continue

        if percent:
            y = 100.0 * y

        y_plot = moving_average(y, smooth_window)

        plt.plot(x, y_plot, linewidth=1.8, label=nice_label(row))
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=180)
    plt.close()


def plot_final_bar(
    summary_rows: List[Dict[str, str]],
    plots_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    percent: bool = False,
) -> None:
    rows = sorted(summary_rows, key=case_sort_key)
    labels = [nice_label(r) for r in rows]
    values = [to_float(r[metric]) for r in rows]

    if percent:
        values = [100.0 * v for v in values]

    plt.figure(figsize=(11, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join("experiments", "sensitivity", "custom_ppo_one_obstacle_random_corrected"),
    )
    parser.add_argument("--task", type=str, default="one_obstacle_random")
    parser.add_argument("--model-name", type=str, default="final_model")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    cases_path = base_dir / "sensitivity_cases.csv"
    cases = read_csv(cases_path)

    if not cases:
        raise FileNotFoundError(f"No cases found at {cases_path}")

    plots_dir = base_dir / "comparative_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for row in cases:
        case_dir = base_dir / row["name"]
        final = read_final_eval(case_dir, args.task, args.model_name)

        out = dict(row)
        out.update({k: str(v) for k, v in final.items()})
        summary_rows.append(out)

    summary_path = base_dir / "sensitivity_summary.csv"
    fieldnames = list(summary_rows[0].keys())

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary: {summary_path}")

    groups = ["learning_rate", "entropy", "ppo_clip", "reward"]

    for group in groups:
        plot_group_curves(
            cases, base_dir, plots_dir, group,
            metric="eval_success_rate",
            ylabel="Evaluation success rate [%]",
            filename=f"success_{group}.png",
            percent=True,
            smooth_window=args.smooth_window,
        )
        plot_group_curves(
            cases, base_dir, plots_dir, group,
            metric="eval_collision_rate",
            ylabel="Evaluation collision rate [%]",
            filename=f"collision_{group}.png",
            percent=True,
            smooth_window=args.smooth_window,
        )
        plot_group_curves(
            cases, base_dir, plots_dir, group,
            metric="eval_return_mean",
            ylabel="Evaluation mean return",
            filename=f"return_{group}.png",
            percent=False,
            smooth_window=args.smooth_window,
        )

    plot_final_bar(
        summary_rows, plots_dir,
        metric="success_rate",
        ylabel="Final success rate [%]",
        filename="final_success_by_case.png",
        percent=True,
    )
    plot_final_bar(
        summary_rows, plots_dir,
        metric="collision_rate",
        ylabel="Final collision rate [%]",
        filename="final_collision_by_case.png",
        percent=True,
    )
    plot_final_bar(
        summary_rows, plots_dir,
        metric="mean_return",
        ylabel="Final mean return",
        filename="final_return_by_case.png",
        percent=False,
    )

    print(f"Wrote plots to: {plots_dir}")
    print()
    print("Quick final summary:")
    for row in sorted(summary_rows, key=case_sort_key):
        print(
            f"{row['name']:<48s} "
            f"success={100*to_float(row['success_rate']):5.1f}% | "
            f"collision={100*to_float(row['collision_rate']):5.1f}% | "
            f"return={to_float(row['mean_return']):8.2f}"
        )


if __name__ == "__main__":
    main()

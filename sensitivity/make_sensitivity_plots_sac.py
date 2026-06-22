"""
Create plots and a summary table for the SAC sensitivity runs.

The script reads sac_sensitivity_cases.csv, training logs, and evaluation CSVs
from each case folder. It then writes sac_sensitivity_summary.csv and the
comparison figures used in the report.

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
        return f"lr={float(row['learning_rate']):g}"
    if group == "entropy":
        return f"entropy={row['ent_coef']}"
    if group == "reward":
        return {
            "reward_weak_guidance": "weak guidance",
            "reward_aggressive_goal": "aggressive goal",
            "reward_weak_safety": "weak safety",
            "reward_safety_focused": "safety focused",
        }.get(row["name"], row["name"].replace("_", " "))

    return row["name"]


def case_sort_key(row: Dict[str, str]):
    group_order = {"baseline": 0, "learning_rate": 1, "entropy": 2, "reward": 3}
    group = row["group"]

    if group == "learning_rate":
        return (group_order[group], float(row["learning_rate"]))
    if group == "entropy":
        return (group_order[group], {"0.01": 1, "auto": 2, "0.1": 3}.get(row["ent_coef"], 99))
    if group == "reward":
        return (
            group_order[group],
            {
                "reward_weak_guidance": 1,
                "reward_aggressive_goal": 2,
                "reward_weak_safety": 3,
                "reward_safety_focused": 4,
            }.get(row["name"], 99),
        )

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


def plot_group_curves(rows, base_dir, plots_dir, group, metric, ylabel, filename, percent=False, smooth_window=1):
    group_rows = [r for r in rows if r["group"] == group or r["group"] == "baseline"]
    group_rows = sorted(group_rows, key=case_sort_key)

    plt.figure(figsize=(8, 4.8))
    plotted = False

    for row in group_rows:
        curve = load_training_curve(base_dir / row["name"])
        x = curve["timesteps"]
        y = curve[metric]

        if len(x) == 0:
            continue

        if percent:
            y = 100.0 * y

        plt.plot(x, moving_average(y, smooth_window), linewidth=1.8, label=nice_label(row))
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


def plot_final_bar(summary_rows, plots_dir, metric, ylabel, filename, percent=False):
    rows = sorted(summary_rows, key=case_sort_key)
    labels = [nice_label(r) for r in rows]
    values = [to_float(r[metric]) for r in rows]

    if percent:
        values = [100.0 * v for v in values]

    plt.figure(figsize=(10.5, 4.8))
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
    parser.add_argument("--base-dir", type=str, default=os.path.join("experiments", "sac_sensitivity_clean"))
    parser.add_argument("--task", type=str, default="three_obstacles_random")
    parser.add_argument("--model-name", type=str, default="final_model")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    cases_path = base_dir / "sac_sensitivity_cases.csv"
    cases = read_csv(cases_path)

    if not cases:
        raise FileNotFoundError(f"No cases found at {cases_path}")

    plots_dir = base_dir / "comparative_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for row in cases:
        final = read_final_eval(base_dir / row["name"], args.task, args.model_name)
        out = dict(row)
        out.update({k: str(v) for k, v in final.items()})
        summary_rows.append(out)

    summary_path = base_dir / "sac_sensitivity_summary.csv"
    fieldnames = list(summary_rows[0].keys())

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary: {summary_path}")

    for group in ["learning_rate", "entropy", "reward"]:
        plot_group_curves(cases, base_dir, plots_dir, group, "eval_success_rate", "Evaluation success rate [%]", f"success_{group}.png", True, args.smooth_window)
        plot_group_curves(cases, base_dir, plots_dir, group, "eval_collision_rate", "Evaluation collision rate [%]", f"collision_{group}.png", True, args.smooth_window)
        plot_group_curves(cases, base_dir, plots_dir, group, "eval_return_mean", "Evaluation mean return", f"return_{group}.png", False, args.smooth_window)

    plot_final_bar(summary_rows, plots_dir, "success_rate", "Final success rate [%]", "final_success_by_case.png", True)
    plot_final_bar(summary_rows, plots_dir, "collision_rate", "Final collision rate [%]", "final_collision_by_case.png", True)
    plot_final_bar(summary_rows, plots_dir, "mean_return", "Final mean return", "final_return_by_case.png", False)

    print(f"Wrote plots to: {plots_dir}")
    print()
    print("Quick final summary:")
    for row in sorted(summary_rows, key=case_sort_key):
        print(
            f"{row['name']:<45s} "
            f"success={100*to_float(row['success_rate']):5.1f}% | "
            f"collision={100*to_float(row['collision_rate']):5.1f}% | "
            f"return={to_float(row['mean_return']):8.2f}"
        )


if __name__ == "__main__":
    main()

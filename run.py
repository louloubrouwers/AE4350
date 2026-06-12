"""
run.py

Interactive launcher for the drone RL project.

Run from the project root:
    py run.py

Menu order is intentionally demo-first:
    1) choose a PyGame demo
    2) choose/evaluate quantitative results

This file supports SAC, SB3 PPO, and custom PPO.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TASKS = [
    "free_fixed",
    "free_jitter_r2",
    "free_jitter_r5",
    "free_jitter_r8",
    "free_random",
    "one_obstacle_fixed",
    "one_obstacle_jitter_r2",
    "one_obstacle_jitter_r4",
    "one_obstacle_jitter_r6",
    "one_obstacle_random",
    "two_obstacles_random",
    "three_obstacles_random",
    "three_obstacles_wind",
]

DEFAULT_SAC_DIR = "experiments/sac/three_obstacles_random"
DEFAULT_SB3_PPO_DIR = "experiments/sb3_ppo/one_obstacle_random_from_free_random"
DEFAULT_CUSTOM_PPO_DIR = "experiments/custom_ppo/one_obstacle_random_final_safety_tuned"


def existing_script(*candidates: str) -> str:
    for candidate in candidates:
        if (ROOT / candidate).exists():
            return candidate
    return candidates[0]


def run(cmd: list[str]) -> None:
    script = cmd[1] if len(cmd) > 1 else ""

    if script.endswith(".py") and not (ROOT / script).exists():
        print("\n" + "=" * 80)
        print("Could not run command because this script was not found:")
        print(f"  {script}")
        print("Check the filename/location, or create the missing demo/evaluation file.")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True, cwd=ROOT)


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ask_int(prompt: str, default: int) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    if not value:
        return str(default)
    int(value)
    return value


def choose_task(default: str, selected_model_dir: str | None = None) -> str:
    print("\nAvailable tasks:")
    for i, task in enumerate(TASKS, start=1):
        marker = "  <-- recommended from selected model" if task == default else ""
        print(f"  {i:2d}. {task}{marker}")

    if selected_model_dir:
        print()
        print(f"Selected model folder: {selected_model_dir}")
        print("Choose the task corresponding to this model folder.")
        print("Press Enter to use the recommended task.")

    value = input(f"Task name or number [{default}]: ").strip()

    if not value:
        return default

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(TASKS):
            return TASKS[idx - 1]
        print(f"Invalid task number {idx}; using default.")
        return default

    if value not in TASKS:
        print(f"Warning: '{value}' is not one of the known tasks. Using it anyway.")
    return value


def infer_task_from_model_dir(model_dir: str, fallback: str) -> str:
    text = model_dir.replace("\\", "/").lower()
    for task in sorted(TASKS, key=len, reverse=True):
        if task.lower() in text:
            return task
    return fallback


def _relative_display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def get_model_dirs(base_folder: str) -> list[Path]:
    base = ROOT / base_folder
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def folder_tags(folder: Path) -> str:
    tags = []
    if (folder / "final_model.zip").exists() or (folder / "final_model.pt").exists():
        tags.append("final_model")
    if (folder / "best_model.zip").exists() or (folder / "best_model.pt").exists():
        tags.append("best_model")
    if (folder / "best_vecnormalize.pkl").exists():
        tags.append("best_vecnormalize")
    if (folder / "final_vecnormalize.pkl").exists():
        tags.append("final_vecnormalize")
    if (folder / "vecnormalize.pkl").exists():
        tags.append("vecnormalize")
    return f" ({', '.join(tags)})" if tags else ""


def list_model_dirs(base_folder: str) -> None:
    print(f"\nAvailable model folders in {base_folder}:")
    folders = get_model_dirs(base_folder)
    if not folders:
        print("  No model folders found.")
        return

    for i, folder in enumerate(folders, start=1):
        folder_text = _relative_display(folder)
        inferred = infer_task_from_model_dir(folder_text, fallback="?")
        inferred_text = f" | suggested task: {inferred}" if inferred != "?" else ""
        print(f"  {i:2d}. {folder_text}{folder_tags(folder)}{inferred_text}")


def ask_model_dir(base_folder: str, default: str) -> str:
    folders = get_model_dirs(base_folder)
    list_model_dirs(base_folder)
    value = input(f"Model directory or number [{default}]: ").strip()

    if not value:
        return default

    if value.isdigit():
        idx = int(value)
        if 1 <= idx <= len(folders):
            return _relative_display(folders[idx - 1])
        print(f"Invalid number {idx}; using default.")
        return default

    return value


def ask_model_and_task(algo_folder: str, default_model_dir: str, fallback_task: str) -> tuple[str, str]:
    model_dir = ask_model_dir(algo_folder, default_model_dir)
    suggested_task = infer_task_from_model_dir(model_dir, fallback_task)
    print(f"\nSuggested task from selected model folder: {suggested_task}")
    task = choose_task(suggested_task, selected_model_dir=model_dir)
    return model_dir, task


def ask_model_name(default: str = "best_model") -> str:
    print("\nModel name options usually are:")
    print("  - best_model")
    print("  - final_model")
    return ask("Model name", default)


def list_sensitivity_dirs(default_base: str) -> None:
    base = ROOT / "experiments"
    print("\nAvailable sensitivity/result folders in experiments:")

    if not base.exists():
        print("  No experiments folder found yet.")
        print(f"  Suggested default: {default_base}")
        return

    folders = [p for p in base.iterdir() if p.is_dir() and "sensitivity" in p.name.lower()]
    if not folders:
        print("  No sensitivity folders found yet.")
        print(f"  Suggested default: {default_base}")
        return

    for i, folder in enumerate(sorted(folders), start=1):
        print(f"  {i:2d}. {_relative_display(folder)}")


def ask_sensitivity_dir(default: str) -> str:
    list_sensitivity_dirs(default)
    return ask("Sensitivity folder", default)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main() -> None:
    py = sys.executable

    print("=" * 80)
    print("Drone RL Project Launcher")
    print("=" * 80)
    print()
    print("PyGame demos for best models:")
    print("  1. Run SAC pygame demo")
    print("  2. Run SB3 PPO pygame demo")
    print("  3. Run custom PPO pygame demo")
    print()
    print("Quantitative evaluation:")
    print("  4. Evaluate SAC")
    print("  5. Evaluate SB3 PPO")
    print("  6. Evaluate custom PPO")
    print()
    print("Plots:")
    print("  7. Generate custom PPO sensitivity plots")
    print("  8. Generate SAC sensitivity plots")
    print()
    print("Training (long):")
    print("  9. Train SAC")
    print(" 10. Train SB3 PPO")
    print(" 11. Train custom PPO")
    print()
    print("Sensitivity experiments (long):")
    print(" 12. Run custom PPO sensitivity")
    print(" 13. Run SAC sensitivity")
    print()

    choice = input("Choose option: ").strip()

    # ------------------------------------------------------------------
    # PyGame demos first
    # ------------------------------------------------------------------

    if choice == "1":
        model_dir, task = ask_model_and_task(
            "experiments/sac",
            DEFAULT_SAC_DIR,
            "three_obstacles_random",
        )
        model_name = ask_model_name("best_model")

        run([
            py, existing_script("demos/demo_sac_pygame.py", "demo_sac_pygame.py"),
            "--difficulty", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
        ])

    elif choice == "2":
        model_dir, task = ask_model_and_task(
            "experiments/sb3_ppo",
            DEFAULT_SB3_PPO_DIR,
            "one_obstacle_random",
        )
        model_name = ask_model_name("best_model")

        run([
            py, existing_script("demos/demo_sb3_ppo_pygame.py", "demo_sb3_ppo_pygame.py"),
            "--difficulty", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
        ])

    elif choice == "3":
        model_dir, task = ask_model_and_task(
            "experiments/custom_ppo",
            DEFAULT_CUSTOM_PPO_DIR,
            "one_obstacle_random",
        )
        model_name = ask_model_name("best_model")

        run([
            py, existing_script("demos/demo_ppo_pygame.py", "demo_ppo_pygame.py"),
            "--difficulty", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
        ])

    # ------------------------------------------------------------------
    # Quantitative evaluation
    # ------------------------------------------------------------------

    elif choice == "4":
        model_dir, task = ask_model_and_task(
            "experiments/sac",
            DEFAULT_SAC_DIR,
            "three_obstacles_random",
        )
        model_name = ask_model_name("best_model")
        episodes = ask_int("Evaluation episodes", 200)
        seed = ask_int("Seed", 123)

        run([
            py, existing_script("scripts/eval_sac.py", "eval_sac.py"),
            "--task", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
            "--episodes", episodes,
            "--plot", "20",
            "--seed", seed,
        ])

    elif choice == "5":
        model_dir, task = ask_model_and_task(
            "experiments/sb3_ppo",
            DEFAULT_SB3_PPO_DIR,
            "one_obstacle_random",
        )
        model_name = ask_model_name("best_model")
        episodes = ask_int("Evaluation episodes", 200)
        seed = ask_int("Seed", 123)

        run([
            py, existing_script("scripts/eval_sb3_ppo.py", "eval_sb3_ppo.py"),
            "--task", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
            "--episodes", episodes,
            "--plot", "20",
            "--seed", seed,
        ])

    elif choice == "6":
        model_dir, task = ask_model_and_task(
            "experiments/custom_ppo",
            DEFAULT_CUSTOM_PPO_DIR,
            "one_obstacle_random",
        )
        model_name = ask_model_name("best_model")
        episodes = ask_int("Evaluation episodes", 200)
        seed = ask_int("Seed", 123)

        run([
            py, existing_script("scripts/eval_ppo.py", "eval_ppo.py"),
            "--task", task,
            "--model-dir", model_dir,
            "--model-name", model_name,
            "--episodes", episodes,
            "--plot", "20",
            "--seed", seed,
        ])

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    elif choice == "7":
        base_dir = ask_sensitivity_dir("experiments/sensitivity_clean")
        task = ask("Task", "one_obstacle_random")
        model_name = ask_model_name("final_model")

        run([
            py, existing_script("sensitivity/make_sensitivity_plots_ppo.py"),
            "--base-dir", base_dir,
            "--task", task,
            "--model-name", model_name,
        ])

    elif choice == "8":
        base_dir = ask_sensitivity_dir("experiments/sac_sensitivity_clean")
        task = ask("Task", "three_obstacles_random")
        model_name = ask_model_name("final_model")

        run([
            py, existing_script("sensitivity/make_sensitivity_plots_sac.py"),
            "--base-dir", base_dir,
            "--task", task,
            "--model-name", model_name,
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    elif choice == "9":
        task = choose_task("three_obstacles_random")
        list_model_dirs("experiments/sac")
        parent = ask("Load from parent checkpoint", "experiments/sac/two_obstacles_random")
        out_dir = ask("Output directory", f"experiments/sac/{task}")
        timesteps = ask_int("Training timesteps", 1_000_000)
        lr = ask("Learning rate", "0.0003")
        ent_coef = ask("Entropy coefficient", "auto")

        run([
            py, existing_script("scripts/train_sac.py", "train_sac.py"),
            "--task", task,
            "--load-from", parent,
            "--total-timesteps", timesteps,
            "--out-dir", out_dir,
            "--learning-rate", lr,
            "--ent-coef", ent_coef,
        ])

    elif choice == "10":
        task = choose_task("one_obstacle_random")
        list_model_dirs("experiments/sb3_ppo")
        parent = ask("Load from parent checkpoint", DEFAULT_SB3_PPO_DIR)
        load_model_name = ask_model_name("best_model")
        out_dir = ask("Output directory", f"experiments/sb3_ppo/{task}")
        timesteps = ask_int("Training timesteps", 1_000_000)
        lr = ask("Learning rate", "0.0003")
        ent_coef = ask("Entropy coefficient", "0.001")

        cmd = [
            py, existing_script("scripts/train_sb3_ppo.py", "train_sb3_ppo.py"),
            "--task", task,
            "--load-from", parent,
            "--total-timesteps", timesteps,
            "--out-dir", out_dir,
            "--learning-rate", lr,
            "--ent-coef", ent_coef,
        ]

        # Keep this if train_sb3_ppo.py includes the VecNormalize fix.
        # Remove the next two entries if your training script does not support it.
        cmd += ["--load-model-name", load_model_name]

        run(cmd)

    elif choice == "11":
        task = choose_task("one_obstacle_random")
        list_model_dirs("experiments/custom_ppo")
        parent = ask("Load from parent checkpoint", "experiments/custom_ppo/free_random_from_jitter_r8_seed30")
        out_dir = ask("Output directory", f"experiments/custom_ppo/{task}")
        timesteps = ask_int("Training timesteps", 1_000_000)
        seed = ask_int("Seed", 500)
        lr = ask("Learning rate", "0.00015")
        entropy = ask("Entropy coefficient", "0.001")
        clip = ask("PPO clip coefficient", "0.2")

        run([
            py, existing_script("scripts/train_ppo.py", "train_ppo.py"),
            "--task", task,
            "--load-from", parent,
            "--total-timesteps", timesteps,
            "--out-dir", out_dir,
            "--seed", seed,
            "--lr", lr,
            "--entropy-coef", entropy,
            "--clip-coef", clip,
        ])

    # ------------------------------------------------------------------
    # Sensitivity experiments
    # ------------------------------------------------------------------

    elif choice == "12":
        list_model_dirs("experiments/custom_ppo")
        parent = ask("Parent checkpoint", "experiments/custom_ppo/free_random_from_jitter_r8_seed30")
        task = ask("Task", "one_obstacle_random")
        timesteps = ask_int("Timesteps per case", 1_000_000)
        out_dir = ask("Output folder", "experiments/sensitivity_clean")

        run([
            py, existing_script("sensitivity/custom_ppo_sensitivity.py"),
            "--parent-dir", parent,
            "--task", task,
            "--timesteps", timesteps,
            "--base-out", out_dir,
        ])

    elif choice == "13":
        list_model_dirs("experiments/sac")
        parent = ask("Parent checkpoint", "experiments/sac/two_obstacles_random")
        task = ask("Task", "three_obstacles_random")
        timesteps = ask_int("Timesteps per case", 500_000)
        out_dir = ask("Output folder", "experiments/sac_sensitivity_clean")

        run([
            py, existing_script("sensitivity/sac_sensitivity.py"),
            "--parent-dir", parent,
            "--task", task,
            "--timesteps", timesteps,
            "--base-out", out_dir,
        ])

    else:
        print("Invalid choice. Please run again and select a number from the menu.")


if __name__ == "__main__":
    main()

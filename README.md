# Reinforcement Learning for 2D Drone Navigation with Obstacle Avoidance

This project implements and compares reinforcement-learning controllers for a two-dimensional drone navigation task with obstacle avoidance. The drone must reach a goal while avoiding walls and rectangular no-fly zones using local ray-based sensing.

The project compares:

- **Soft Actor-Critic (SAC)** from Stable-Baselines3
- **PPO** from Stable-Baselines3
- **Custom PPO**, a self-implemented PPO controller

The easiest way to run the project is through the interactive launcher:

```bash
py run.py
```

or:

```bash
python run.py
```

The launcher provides recommended PyGame demonstrations, flexible evaluation for any trained model and task, training commands, sensitivity experiments, and sensitivity plot generation.

---

## 1. Installation

Create and activate a virtual environment if desired, then install the requirements:

```bash
pip install -r requirements.txt
```

If the Stable-Baselines3 extra dependencies fail to install, the base package is sufficient for this project:

```bash
pip install stable-baselines3
```

Run all commands from the project root folder.

---

## 2. Project structure

```text
Final version/
│
├── run.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── drone_env.py
│   ├── env_configs.py
│   ├── wrappers.py
│   └── eval_common.py
│
├── custom_ppo/
│   ├── __init__.py
│   ├── custom_ppo_agent.py
│   ├── ppo_networks.py
│   └── ppo_rollout_buffer.py
│
├── scripts/
│   ├── __init__.py
│   ├── train_sac.py
│   ├── train_sb3_ppo.py
│   ├── train_ppo.py
│   ├── eval_sac.py
│   ├── eval_sb3_ppo.py
│   ├── eval_ppo.py
│   └── evaluate_all_existing_models.py
│
├── sensitivity/
│   ├── __init__.py
│   ├── custom_ppo_sensitivity.py
│   ├── sac_sensitivity.py
│   ├── make_sensitivity_plots_ppo.py
│   └── make_sensitivity_plots_sac.py
│
├── demos/
│   ├── __init__.py
│   ├── demo_sac_pygame.py
│   ├── demo_sb3_ppo_pygame.py
│   └── demo_ppo_pygame.py
│
└── experiments/
    ├── sac/
    ├── sb3_ppo/
    ├── custom_ppo/
    ├── sensitivity_clean/
    └── sac_sensitivity_1500k/
```

Folder roles:

```text
core/          Shared environment, task definitions, wrappers, and evaluation helpers
custom_ppo/    Self-implemented PPO algorithm
scripts/       Training and evaluation scripts
sensitivity/   Sensitivity experiment and plotting scripts
demos/         PyGame visualization demos
experiments/   Trained models, logs, evaluation CSVs, and generated plots
```

---

## 3. Environment and tasks

The environment is a two-dimensional continuous-control drone navigation problem. The drone observes:

- its position and velocity,
- the goal position relative to the drone,
- local ray-based distances to nearby walls and obstacles.

The action is a continuous two-dimensional acceleration command.

All final task definitions, obstacle layouts, reward settings, and aliases are centralized in:

```text
core/env_configs.py
```

The final task names are:

```text
free_fixed
free_jitter_r2
free_jitter_r5
free_jitter_r8
free_random
one_obstacle_fixed
one_obstacle_jitter_r2
one_obstacle_jitter_r4
one_obstacle_jitter_r6
one_obstacle_random
two_obstacles_random
three_obstacles_random
three_obstacles_wind
```

The main final benchmark is:

```text
three_obstacles_random
```

The wind robustness task is:

```text
three_obstacles_wind
```

The one-obstacle curriculum uses a consistent obstacle size for the fixed, jittered, and random one-obstacle tasks:

```python
Obstacle(9.0, 8.5, 2.0, 3.0)
```

---

## 4. Reward settings

All algorithms use the same final reward settings for fair comparison:

```python
r_goal = 350.0
r_collision = -100.0
r_step = -0.01
r_action = -0.001
r_progress = 4.0
r_proximity = -0.01
proximity_threshold = 1.0
```

The reward function encourages the drone to reach the goal, make progress during the episode, avoid collisions, avoid excessive control inputs, and maintain a safety margin near obstacles and walls.

For reward-sensitivity analysis, reward terms can be overridden using environment variables:

```text
DRONE_R_GOAL
DRONE_R_COLLISION
DRONE_R_STEP
DRONE_R_ACTION
DRONE_R_PROGRESS
DRONE_R_PROXIMITY
DRONE_PROXIMITY_THRESHOLD
```

---

## 5. Interactive launcher

Run the launcher from the project root:

```bash
py run.py
```

The menu is organized by workflow:

```text
Recommended PyGame demos:
  1. Run SAC demo on three_obstacles_random
  2. Run SAC demo on three_obstacles_wind
  3. Run SB3 PPO demo on one_obstacle_random
  4. Run custom PPO demo on one_obstacle_random

Evaluation, any trained model/task:
  5. Evaluate any SAC model on any task
  6. Evaluate any SB3 PPO model on any task
  7. Evaluate any custom PPO model on any task

Training (long):
  8. Train SAC
  9. Train SB3 PPO
 10. Train custom PPO

Sensitivity experiments (long):
 11. Run custom PPO sensitivity
 12. Run SAC sensitivity

Sensitivity plots:
 13. Generate custom PPO sensitivity plots
 14. Generate SAC sensitivity plots
```

The recommended PyGame demos use sensible task/model combinations for presentation:

- SAC is demonstrated on the final three-obstacle task.
- SAC can also be demonstrated on the wind robustness task.
- SB3 PPO and custom PPO are demonstrated on the one-obstacle random task, where they learned more reliable obstacle-avoidance behaviour.

The evaluation options are more flexible: they allow any trained model folder to be evaluated on any task. This is useful for reproducing final tables, checking intermediate curriculum models, or debugging.

---


## 6. Direct evaluation commands

The launcher is recommended, but the scripts can also be run directly.

### SAC on the final task

```bash
py scripts/eval_sac.py --task three_obstacles_random --model-dir experiments/sac/three_obstacles_random --model-name best_model --episodes 200 --plot 20 --seed 123
```

### SAC wind robustness

```bash
py scripts/eval_sac.py --task three_obstacles_wind --model-dir experiments/sac/three_obstacles_wind_from_random --model-name best_model --episodes 200 --plot 20 --seed 123
```

### SB3 PPO

```bash
py scripts/eval_sb3_ppo.py --task one_obstacle_random --model-dir experiments/sb3_ppo/one_obstacle_random_from_free_random --model-name best_model --episodes 200 --plot 20 --seed 123
```

### Custom PPO

```bash
py scripts/eval_ppo.py --task one_obstacle_random --model-dir experiments/custom_ppo/one_obstacle_random_final_lrlow_ent003 --model-name best_model --episodes 200 --plot 20 --seed 123
```

The evaluation scripts report:

```text
success rate
collision rate
mean return
mean number of steps
mean path length
```

They also save evaluation CSV files and trajectory plots.

---

## 7. Direct training commands

Training can take a long time. The commands below reproduce the main training style used in the project.

### SAC

```bash
py scripts/train_sac.py --task three_obstacles_random --load-from experiments/sac/two_obstacles_random --total-timesteps 1000000 --out-dir experiments/sac/three_obstacles_random --learning-rate 0.0003 --ent-coef auto
```

### SAC wind fine-tuning

```bash
py scripts/train_sac.py --task three_obstacles_wind --load-from experiments/sac/three_obstacles_random --total-timesteps 1000000 --out-dir experiments/sac/three_obstacles_wind_from_random --learning-rate 0.0003 --ent-coef auto
```

### SB3 PPO

```bash
py scripts/train_sb3_ppo.py --task one_obstacle_random --load-from experiments/sb3_ppo/one_obstacle_random_from_free_random --total-timesteps 1000000 --out-dir experiments/sb3_ppo/one_obstacle_random_from_free_random --learning-rate 0.0003 --ent-coef 0.001 --load-model-name best_model
```

### Custom PPO

```bash
py scripts/train_ppo.py --task one_obstacle_random --load-from experiments/custom_ppo/free_random_from_r5 --total-timesteps 1000000 --out-dir experiments/custom_ppo/one_obstacle_random_final_lrlow_ent003 --seed 500 --lr 0.00015 --entropy-coef 0.003 --clip-coef 0.2
```

---

## 8. Suggested curriculum

The general SAC curriculum is:

```text
free_random
→ one_obstacle_fixed
→ one_obstacle_jitter_r2
→ one_obstacle_jitter_r4
→ one_obstacle_jitter_r6
→ one_obstacle_random
→ two_obstacles_random
→ three_obstacles_random
→ three_obstacles_wind
```

For SB3 PPO and custom PPO, the same task family can be used, but these methods are less reliable on the hardest multi-obstacle tasks. The most relevant final demonstration for the PPO variants is therefore:

```text
one_obstacle_random
```

The custom PPO model used in the report is the low-learning-rate one-obstacle model:

```text
experiments/custom_ppo/one_obstacle_random_final_lrlow_ent003
```

---

## 9. Sensitivity analysis

Sensitivity scripts are located in:

```text
sensitivity/
```

### Custom PPO sensitivity

Task:

```text
one_obstacle_random
```

Parent checkpoint:

```text
experiments/custom_ppo/free_random_from_r5
```

Run:

```bash
py sensitivity/custom_ppo_sensitivity.py --parent-dir experiments/custom_ppo/free_random_from_r5 --task one_obstacle_random --timesteps 1000000 --base-out experiments/sensitivity_clean
```

Generate plots:

```bash
py sensitivity/make_sensitivity_plots_ppo.py --base-dir experiments/sensitivity_clean --task one_obstacle_random --model-name best_model
```

Tested parameter groups:

```text
learning rate
entropy coefficient
PPO clipping coefficient
reward design
```

### SAC sensitivity

Task:

```text
three_obstacles_random
```

Parent checkpoint:

```text
experiments/sac/two_obstacles_random
```

Run:

```bash
py sensitivity/sac_sensitivity.py --parent-dir experiments/sac/two_obstacles_random --task three_obstacles_random --timesteps 1500000 --base-out experiments/sac_sensitivity_1500k
```

Generate plots:

```bash
py sensitivity/make_sensitivity_plots_sac.py --base-dir experiments/sac_sensitivity_1500k --task three_obstacles_random --model-name best_model
```

Tested parameter groups:

```text
learning rate
entropy coefficient
reward design
```

---

## 10. Generated outputs

Training scripts generate model checkpoints and logs, for example:

```text
best_model.zip or best_model.pt
final_model.zip or final_model.pt
training_log.csv
learning_curve_success.png
learning_curve_return.png
learning_curve_collision.png
```

Stable-Baselines3 runs may also generate normalization files:

```text
best_vecnormalize.pkl
final_vecnormalize.pkl
vecnormalize.pkl
```

Evaluation scripts generate:

```text
evaluation_<task>_<model_name>.csv
trajectories_<task>_<model_name>.png
```

Sensitivity scripts generate summaries and comparative plots, for example:

```text
sensitivity_summary.csv
sac_sensitivity_summary.csv
comparative_plots/
```

---

## 11. Notes on model-task matching

The recommended demos intentionally use fixed model-task combinations. This avoids showing a model on a task for which it was not trained. For example, custom PPO is demonstrated on `one_obstacle_random`, while SAC is demonstrated on `three_obstacles_random`.

The evaluation options are intentionally more flexible, because they are meant for checking any trained model on any task. When using flexible evaluation, choose a task that matches the selected model folder unless the goal is explicitly to test transfer or robustness.

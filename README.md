# FINAL VERSION - clean RL drone experiments

This folder removes duplicated environment configuration logic from the old files.

## Single sources of truth

- `env_configs.py`: task definitions, obstacles, reward settings, and clean aliases.
- `wrappers.py`: shared `DirectionalAvoidanceWrapper` and `make_env()`.
- `eval_common.py`: shared evaluation, CSV saving, summary printing, trajectory plotting.

## Important design choice

The one-obstacle curriculum now uses the same obstacle size for fixed, jittered, and random tasks:

```python
Obstacle(9.0, 8.5, 2.0, 3.0)
```

This is the old `smallobstaclebaby` size. The old large single obstacle is kept only as optional `one_tall_obstacle_random`.

## Clean task names

Recommended final names:

```text
free_fixed
free_jitter_r2
free_jitter_r5
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

Final target:

```text
three_obstacles_random
```

## Suggested final curriculum

For SAC and SB3 PPO:

```text
free_random
→ one_obstacle_fixed
→ one_obstacle_jitter_r2
→ one_obstacle_jitter_r4
→ one_obstacle_jitter_r6
→ one_obstacle_random
→ two_obstacles_random
→ three_obstacles_random
```

For custom PPO, use the same tasks but it is acceptable if it only reaches `one_obstacle_random` robustly.
# Bio-Inspired Drone Navigation with Reinforcement Learning

This project compares reinforcement-learning controllers for a 2D drone navigation task with obstacle avoidance. The drone must reach a target while avoiding walls and rectangular no-fly zones using local ray-based sensing.

The compared controllers are:

- Soft Actor-Critic (SAC) from Stable-Baselines3
- PPO from Stable-Baselines3
- A self-implemented custom PPO controller

The easiest way to run the project is with the interactive launcher:

```bash
py run.py
```

or:

```bash
python run.py
```

The launcher allows the user to evaluate trained models, run pygame demos, train models, generate sensitivity plots, and launch sensitivity experiments without remembering long command-line commands.

---

## 1. Installation

Create and activate a virtual environment if desired, then install:

```bash
pip install -r requirements.txt
```

If the optional Stable-Baselines3 extras fail to install, the base package is enough for this project:

```bash
pip install stable-baselines3
```

---

## 2. Recommended project structure

```text
Final version/
│
├── run.py
├── requirements.txt
├── README_CLEAN_FINAL.md
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
│   └── eval_ppo.py
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
│   └── demo_ppo_pygame.py
│
└── experiments/
    ├── sac/
    ├── sb3_ppo/
    ├── custom_ppo/
    ├── sensitivity_clean/
    └── sac_sensitivity_clean/
```

Folder roles:

```text
core/          shared environment, task definitions, wrappers, evaluation helpers
custom_ppo/    self-implemented PPO algorithm files
scripts/       training and evaluation scripts
sensitivity/   sensitivity experiments and comparative plotting scripts
demos/         pygame visualization demos
experiments/   generated models, logs, evaluation CSVs, and plots
```

---

## 3. Environment and tasks

The environment is a 2D continuous-control drone navigation problem. The drone observes its own state, the goal direction, and local ray-based distances to nearby walls/obstacles. The action is a continuous 2D acceleration command.

The final task definitions are in:

```text
core/env_configs.py
```

Main tasks:

```text
free_fixed
free_jitter_r2
free_jitter_r5
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

The main final target task is:

```text
three_obstacles_random
```

The wind robustness task is:

```text
three_obstacles_wind
```

---

## 4. Reward settings

The final shared reward setup is:

```python
r_goal = 350.0
r_collision = -100.0
r_step = -0.01
r_action = -0.001
r_progress = 4.0
r_proximity = -0.01
proximity_threshold = 1.0
```

All algorithms use the same final reward settings for fair comparison.

For reward sensitivity analysis, the reward terms can be overridden using environment variables:

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

## 5. Easy use with `run.py`

From the root folder:

```bash
py run.py
```

Menu options:

```text
1. Evaluate final SAC
2. Evaluate final SB3 PPO
3. Evaluate final custom PPO
4. Run SAC pygame demo
5. Run custom PPO pygame demo
6. Generate custom PPO sensitivity plots
7. Generate SAC sensitivity plots
8. Train SAC
9. Train SB3 PPO
10. Train custom PPO
11. Run custom PPO sensitivity
12. Run SAC sensitivity
```

For a quick demonstration, choose:

```text
1. Evaluate final SAC
3. Evaluate final custom PPO
4. Run SAC pygame demo
5. Run custom PPO pygame demo
```

---

## 6. Direct evaluation commands

### SAC

```bash
py scripts/eval_sac.py --task three_obstacles_random --model-dir experiments/sac/three_obstacles_random --model-name final_model --episodes 200 --plot 20 --seed 123
```

### SAC wind robustness

```bash
py scripts/eval_sac.py --task three_obstacles_wind --model-dir experiments/sac/three_obstacles_wind_from_random --model-name final_model --episodes 200 --plot 20 --seed 123
```

### SB3 PPO

```bash
py scripts/eval_sb3_ppo.py --task three_obstacles_random --model-dir experiments/sb3_ppo/three_obstacles_random --model-name final_model --episodes 200 --plot 20 --seed 123
```

### Custom PPO

```bash
py scripts/eval_ppo.py --task one_obstacle_random --model-dir experiments/custom_ppo/one_obstacle_random_final_lrlow_ent003 --model-name final_model --episodes 200 --plot 20 --seed 900
```

---

## 7. Direct training commands

### SAC

```bash
py scripts/train_sac.py --task three_obstacles_random --load-from experiments/sac/two_obstacles_random --total-timesteps 1000000 --out-dir experiments/sac/three_obstacles_random --learning-rate 0.0003 --ent-coef auto
```

### SB3 PPO

```bash
py scripts/train_sb3_ppo.py --task three_obstacles_random --load-from experiments/sb3_ppo/two_obstacles_random --total-timesteps 1000000 --out-dir experiments/sb3_ppo/three_obstacles_random --learning-rate 0.0003
```

### Custom PPO

```bash
py scripts/train_ppo.py --task one_obstacle_random --load-from experiments/custom_ppo/free_random_from_r5 --total-timesteps 1000000 --out-dir experiments/custom_ppo/one_obstacle_random_final_lrlow_ent003 --seed 500 --lr 0.00015 --entropy-coef 0.003 --clip-coef 0.2
```

---

## 8. Sensitivity analysis

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

Generate comparative plots:

```bash
py sensitivity/make_sensitivity_plots_ppo.py --base-dir experiments/sensitivity_clean --task one_obstacle_random --model-name final_model
```

Tested parameters:

```text
learning rate: 0.00015 / 0.0003 / 0.001
entropy coefficient: 0.001 / 0.003 / 0.008
PPO clipping coefficient: 0.05 / 0.2 / 0.4
reward design: baseline, weak guidance, aggressive goal, weak safety, safety focused
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
py sensitivity/sac_sensitivity.py --parent-dir experiments/sac/two_obstacles_random --task three_obstacles_random --timesteps 500000 --base-out experiments/sac_sensitivity_clean
```

Generate comparative plots:

```bash
py sensitivity/make_sensitivity_plots_sac.py --base-dir experiments/sac_sensitivity_clean --task three_obstacles_random --model-name final_model
```

Tested parameters:

```text
learning rate: 0.0001 / 0.0003 / 0.001
entropy coefficient: 0.01 / auto / 0.1
reward design: baseline, weak guidance, aggressive goal, weak safety, safety focused
```

---

## 9. Generated outputs

Training scripts generate:

```text
best_model or best_model.zip
final_model or final_model.zip
training_log.csv
learning_curve_success.png
learning_curve_return.png
learning_curve_collision.png
```

Evaluation scripts generate:

```text
evaluation_<task>_<model_name>.csv
trajectories_<task>_<model_name>.png
```

Sensitivity plot scripts generate:

```text
sensitivity_summary.csv
sac_sensitivity_summary.csv
comparative_plots/
```

---

## 10. Reproducibility notes

Small differences between runs are expected because reinforcement learning and randomized start-goal evaluation are stochastic.

For fair comparison:

- all algorithms use the same environment and reward settings
- the same task definitions are used for training and evaluation
- evaluation reports success rate, collision rate, mean return, mean episode length, and mean path length
- sensitivity analysis varies one main parameter or reward-design setting at a time

---

## 11. Troubleshooting

### Import errors after reorganizing folders

Run scripts from the project root, for example:

```bash
py run.py
```

or:

```bash
py scripts/eval_sac.py --help
```

### Stable-Baselines3 installation

If this fails:

```bash
pip install "stable-baselines3[extra]"
```

use:

```bash
pip install stable-baselines3
```

The base package is sufficient for this project.

### Gym warning

Some dependencies may print a warning about Gym being unmaintained. This can usually be ignored if the scripts run normally.

---

## 12. Suggested quick grading demo

Run:

```bash
py run.py
```

Then choose:

```text
1. Evaluate final SAC
3. Evaluate final custom PPO
4. Run SAC pygame demo
5. Run custom PPO pygame demo
```

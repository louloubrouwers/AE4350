"""
Pygame visualization for a trained Stable-Baselines3 PPO drone policy.

Can be run through run.py 
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.drone_env import DroneNavEnv, EnvConfig, Obstacle
from core.env_configs import make_config as make_base_config, task_choices

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_pair(values):
    if values is None:
        return None
    return float(values[0]), float(values[1])


def make_demo_config(task: str, start: Optional[Tuple[float, float]] = None, goal: Optional[Tuple[float, float]] = None) -> EnvConfig:
    """Create a demo config from the shared final task definitions."""
    cfg = make_base_config(task)

    if start is not None:
        cfg.fixed_start = start
        cfg.jitter_start_center = None

    if goal is not None:
        cfg.fixed_goal = goal
        cfg.jitter_goal_center = None

    return cfg


class Mapper:
    def __init__(self, world_size: float, screen_size: int, margin: int):
        self.world_size = float(world_size)
        self.screen_size = int(screen_size)
        self.margin = int(margin)
        self.plot_size = self.screen_size - 2 * self.margin
        self.scale = self.plot_size / self.world_size

    def point(self, p):
        x = self.margin + float(p[0]) * self.scale
        y = self.margin + (self.world_size - float(p[1])) * self.scale
        return int(round(x)), int(round(y))

    def rect(self, ob: Obstacle):
        x, y_top = self.point((ob.x, ob.y + ob.h))
        return pygame.Rect(
            x,
            y_top,
            int(round(ob.w * self.scale)),
            int(round(ob.h * self.scale)),
        )

    def length(self, value: float) -> int:
        return int(round(float(value) * self.scale))


def draw_arrow(surface, color, start_px, vector, scale=25.0, width=3):
    vx, vy = float(vector[0]), float(vector[1])
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-8:
        return

    end_px = (
        int(start_px[0] + scale * vx / max(norm, 1.0)),
        int(start_px[1] - scale * vy / max(norm, 1.0)),
    )
    pygame.draw.line(surface, color, start_px, end_px, width)


def draw_rays(surface, env: DroneNavEnv, mapper: Mapper):
    if not hasattr(env, "_cast_rays"):
        return

    origin = env.pos
    origin_px = mapper.point(origin)
    angles = np.linspace(0.0, 2.0 * np.pi, env.cfg.n_rays, endpoint=False)
    dists = env._cast_rays()

    for angle, dist in zip(angles, dists):
        end = origin + np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(dist)
        pygame.draw.line(surface, (240, 220, 60), origin_px, mapper.point(end), 1)


def draw_wind_field(screen, mapper: Mapper, env: DroneNavEnv):
    """
    Visualize wind/gusts as light-blue arrows.

    If the task has non-zero mean wind, arrows tend to point in that direction.
    If the task has zero-mean stochastic wind, arrows show changing random gusts.
    """
    wind_mean = np.asarray(getattr(env.cfg, "wind_mean", (0.0, 0.0)), dtype=float)
    wind_std = float(getattr(env.cfg, "wind_std", 0.0))

    if np.linalg.norm(wind_mean) < 1e-8 and wind_std <= 1e-8:
        return

    # Update roughly four times per second so gusts move but do not flicker too much.
    rng = np.random.default_rng(int(pygame.time.get_ticks() // 250))

    grid_x = np.linspace(3.0, env.cfg.world_size - 3.0, 5)
    grid_y = np.linspace(3.0, env.cfg.world_size - 3.0, 5)

    for x in grid_x:
        for y in grid_y:
            gust = wind_mean.copy()

            if wind_std > 0.0:
                gust += rng.normal(0.0, wind_std, size=2)

            norm = np.linalg.norm(gust)
            if norm < 1e-6:
                continue

            start_px = mapper.point((x, y))
            # Scale is chosen only for visualization; it does not affect physics.
            end_world = np.array([x, y]) + 1.2 * gust / max(norm, 1.0)
            end_px = mapper.point(end_world)

            pygame.draw.line(screen, (120, 210, 255), start_px, end_px, 2)
            pygame.draw.circle(screen, (120, 210, 255), end_px, 3)


def draw_background(screen, mapper: Mapper, panel_width: int):
    # Environment background only occupies the square map area.
    screen.fill((22, 25, 30))
    map_surface = pygame.Rect(0, 0, mapper.screen_size, mapper.screen_size)
    pygame.draw.rect(screen, (34, 62, 43), map_surface)

    boundary = pygame.Rect(mapper.margin, mapper.margin, mapper.plot_size, mapper.plot_size)
    pygame.draw.rect(screen, (38, 76, 49), boundary)

    road_color = (48, 52, 58)
    road_edge = (78, 82, 88)
    streets = [
        ("h", 2, 1.3),
        ("h", 17, 0.9),
        ("v", 3.4, 1.0),
        ("v", 16.8, 1.1),
    ]

    for kind, center, width in streets:
        if kind == "h":
            p1 = mapper.point((0.0, center + width / 2))
            p2 = mapper.point((mapper.world_size, center - width / 2))
            rect = pygame.Rect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
        else:
            p1 = mapper.point((center - width / 2, mapper.world_size))
            p2 = mapper.point((center + width / 2, 0.0))
            rect = pygame.Rect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])

        pygame.draw.rect(screen, road_color, rect)
        pygame.draw.rect(screen, road_edge, rect, 1)

        if kind == "h":
            y = rect.centery
            for x in range(rect.left + 10, rect.right - 10, 28):
                pygame.draw.line(screen, (150, 150, 130), (x, y), (x + 14, y), 1)
        else:
            x = rect.centerx
            for y in range(rect.top + 10, rect.bottom - 10, 28):
                pygame.draw.line(screen, (150, 150, 130), (x, y), (x, y + 14), 1)

    tree_world = [
        (2.0, 3.0), (3.2, 3.8), (16.8, 17.2), (18.0, 16.4),
        (2.0, 17.4), (17.6, 2.2), (10.3, 2.0), (7.0, 17.8),
        (18.2, 8.0), (1.4, 8.8)
    ]
    for p in tree_world:
        px = mapper.point(p)
        pygame.draw.circle(screen, (28, 120, 55), px, 8)
        pygame.draw.circle(screen, (55, 155, 75), (px[0] - 3, px[1] - 3), 5)
        pygame.draw.circle(screen, (80, 55, 30), (px[0], px[1] + 7), 2)

    pygame.draw.rect(screen, (230, 230, 230), boundary, 2)

    # Right-side status panel background.
    panel_rect = pygame.Rect(mapper.screen_size, 0, panel_width, mapper.screen_size)
    pygame.draw.rect(screen, (15, 18, 22), panel_rect)
    pygame.draw.line(screen, (80, 85, 92), (mapper.screen_size, 0), (mapper.screen_size, mapper.screen_size), 2)


def draw_building(screen, rect: pygame.Rect):
    pygame.draw.rect(screen, (82, 88, 98), rect)
    pygame.draw.rect(screen, (190, 195, 205), rect, 2)

    if rect.width > 22 and rect.height > 22:
        step_x = max(13, rect.width // 4)
        step_y = max(13, rect.height // 5)
        for x in range(rect.left + 7, rect.right - 5, step_x):
            for y in range(rect.top + 7, rect.bottom - 5, step_y):
                pygame.draw.rect(screen, (135, 150, 165), pygame.Rect(x, y, 4, 4))


def draw_drone(screen, drone_px):
    body_r = 8
    arm = 18
    rotor_r = 6

    pygame.draw.line(screen, (125, 175, 230), (drone_px[0] - arm, drone_px[1]), (drone_px[0] + arm, drone_px[1]), 3)
    pygame.draw.line(screen, (125, 175, 230), (drone_px[0], drone_px[1] - arm), (drone_px[0], drone_px[1] + arm), 3)

    for rotor in [
        (drone_px[0] - arm, drone_px[1]),
        (drone_px[0] + arm, drone_px[1]),
        (drone_px[0], drone_px[1] - arm),
        (drone_px[0], drone_px[1] + arm),
    ]:
        pygame.draw.circle(screen, (70, 130, 210), rotor, rotor_r)
        pygame.draw.circle(screen, (230, 245, 255), rotor, rotor_r, 1)

    pygame.draw.circle(screen, (80, 150, 255), drone_px, body_r)
    pygame.draw.circle(screen, (235, 245, 255), drone_px, body_r, 2)


def draw_scene(screen, font, env, mapper, episode_return, paused, label, last_action, panel_width: int, show_wind: bool):
    draw_background(screen, mapper, panel_width)

    if show_wind:
        draw_wind_field(screen, mapper, env)

    for ob in env.cfg.obstacles:
        draw_building(screen, mapper.rect(ob))

    if len(env.trajectory) > 1:
        pts = [mapper.point(p) for p in env.trajectory]
        pygame.draw.lines(screen, (250, 250, 250), False, pts, 3)

    draw_rays(screen, env, mapper)

    if len(env.trajectory) > 0:
        start_px = mapper.point(env.trajectory[0])
        pygame.draw.circle(screen, (175, 120, 255), start_px, 6)
        pygame.draw.circle(screen, (240, 220, 255), start_px, 6, 1)

    goal_px = mapper.point(env.goal)
    goal_r = max(5, mapper.length(env.cfg.goal_radius))
    pygame.draw.circle(screen, (70, 220, 100), goal_px, goal_r)
    pygame.draw.circle(screen, (220, 255, 220), goal_px, goal_r, 2)
    pygame.draw.line(screen, (220, 255, 220), goal_px, (goal_px[0], goal_px[1] - 25), 2)
    pygame.draw.polygon(screen, (70, 220, 100), [
        (goal_px[0], goal_px[1] - 25),
        (goal_px[0] + 20, goal_px[1] - 18),
        (goal_px[0], goal_px[1] - 12),
    ])

    drone_px = mapper.point(env.pos)
    draw_drone(screen, drone_px)

    draw_arrow(screen, (80, 210, 255), drone_px, env.vel, scale=35, width=3)
    if last_action is not None:
        draw_arrow(screen, (255, 120, 80), drone_px, last_action, scale=30, width=2)

    dist_goal = float(np.linalg.norm(env.goal - env.pos))
    min_ray = float(np.min(env._cast_rays())) if hasattr(env, "_cast_rays") else 0.0

    wind_mean = getattr(env.cfg, "wind_mean", (0.0, 0.0))
    wind_std = getattr(env.cfg, "wind_std", 0.0)

    panel_x = mapper.screen_size + 14
    y = 16

    lines = [
        label,
        "",
        f"steps: {env.steps}/{env.cfg.max_steps}",
        f"return: {episode_return:+.1f}",
        f"distance to goal: {dist_goal:.2f}",
        f"min ray: {min_ray:.2f}",
        f"wind mean/std: {wind_mean} / {wind_std}",
        f"start/goal: {'manual' if env.cfg.fixed_start is not None and env.cfg.fixed_goal is not None else 'random'}",
        f"paused: {'yes' if paused else 'no'}",
        "",
        "Legend:",
        "yellow rays = distance sensors",
        "orange arrow = action",
        "blue arrow = velocity",
        "light-blue arrows = wind/gusts",
        "purple dot = start",
        "green flag = goal",
        "",
        "Controls:",
        "R = reset same seed",
        "N = new random start/goal",
        "SPACE = pause/resume",
        "Q/ESC = quit",
    ]

    for line in lines:
        img = font.render(line, True, (245, 245, 245))
        screen.blit(img, (panel_x, y))
        y += 18


def run_pygame_loop(args, env: DroneNavEnv, policy_fn, title: str, show_wind: bool = False):
    obs, info = env.reset(seed=args.seed)
    episode_return = 0.0
    done = False
    paused = False
    last_action = None
    final_status = ""

    pygame.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((args.screen_size + args.panel_width, args.screen_size))
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    mapper = Mapper(env.cfg.world_size, args.screen_size, args.margin)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, info = env.reset(seed=args.seed)
                    episode_return = 0.0
                    done = False
                    final_status = ""
                elif event.key == pygame.K_n:
                    obs, info = env.reset()
                    episode_return = 0.0
                    done = False
                    final_status = ""

        if not paused and not done:
            for _ in range(max(1, args.steps_per_frame)):
                action = np.asarray(policy_fn(obs), dtype=np.float32).reshape(2)
                last_action = action.copy()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)

                if terminated or truncated:
                    done = True
                    if info.get("reached_goal", False):
                        final_status = "SUCCESS"
                    elif info.get("collision", False):
                        final_status = "COLLISION"
                    else:
                        final_status = "TIME LIMIT"
                    break

        label = f"{args.algorithm_label} | {args.task} | {args.model_name}"
        if final_status:
            label += f" | {final_status}"

        draw_scene(screen, font, env, mapper, episode_return, paused, label, last_action, args.panel_width, show_wind)
        pygame.display.flip()
        clock.tick(args.fps)

    env.close()
    pygame.quit()


def base_parser(default_task: str, default_model_name: str = "best_model") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "--difficulty", dest="task", choices=task_choices(), default=default_task)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screen-size", type=int, default=850)
    parser.add_argument("--panel-width", type=int, default=340)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    return parser


def load_policy(model_dir: str, model_name: str, cfg: EnvConfig):
    if not model_name.endswith(".zip"):
        model_name += ".zip"

    model_path = os.path.join(model_dir, model_name)
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    dummy_env = DummyVecEnv([lambda: DroneNavEnv(config=cfg)])
    vecnorm = None
    if os.path.exists(vecnorm_path):
        vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
    else:
        print(f"Warning: VecNormalize file not found: {vecnorm_path}. Using raw observations.")

    model = PPO.load(model_path)
    return model, vecnorm


def main():
    parser = base_parser(default_task="one_obstacle_random", default_model_name="best_model")
    args = parser.parse_args()
    args.algorithm_label = "SB3 PPO"

    cfg = make_demo_config(args.task, start=parse_pair(args.start), goal=parse_pair(args.goal))
    env = DroneNavEnv(config=cfg)
    model, vecnorm = load_policy(args.model_dir, args.model_name, cfg)

    def policy_fn(obs):
        obs_arr = np.asarray(obs, dtype=np.float32)[None, :]
        if vecnorm is not None:
            obs_arr = vecnorm.normalize_obs(obs_arr)
        action, _ = model.predict(obs_arr, deterministic=True)
        return action

    run_pygame_loop(args, env, policy_fn, title="SB3 PPO Drone Navigation Demo", show_wind=False)


if __name__ == "__main__":
    main()

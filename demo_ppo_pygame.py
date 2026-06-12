"""
Pygame visualization for the custom PPO drone-navigation policy.

This script reuses the exact PPO environment factory from train_ppo.py, so the
same curriculum names/wrappers/reward shaping are used as in PPO training.

Examples:
    py demo_ppo_pygame.py --difficulty singlefixed --model-dir out/ppo_h128_singlefixed_v1 --model-name best_model

    py demo_ppo_pygame.py --difficulty singlejitter2 --model-dir out/ppo_h128_singlejitter2_stable_v1 --model-name best_model

Manual start/goal:
    py demo_ppo_pygame.py --difficulty singlejitter2 --model-dir out/ppo_h128_singlejitter2_stable_v1 --model-name best_model --start 3 10 --goal 17 10

Keys:
    R       reset same seed
    N       reset with new random start/goal
    SPACE   pause/resume
    ESC/Q   quit
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pygame
import torch

from custom_ppo.custom_ppo_agent import PPOAgent
from train_ppo import RunningNormalizer, make_env
from drone_env import DroneNavEnv, Obstacle


def parse_pair(values):
    if values is None:
        return None
    return float(values[0]), float(values[1])


def load_agent_and_normalizer(model_path: str, device: str = "cpu"):
    payload = torch.load(model_path, map_location=device, weights_only=False)
    agent = PPOAgent.load(model_path, device=device)

    obs_normalizer = None
    extra = payload.get("extra", {}) or {}
    if "obs_normalizer" in extra:
        obs_normalizer = RunningNormalizer((int(payload["state_dim"]),))
        obs_normalizer.load_state_dict(extra["obs_normalizer"])

    return agent, obs_normalizer


def load_policy(model_dir: str, model_name: str, device: str = "cpu"):
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    return load_agent_and_normalizer(model_path, device=device)


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
    origin = env.pos
    origin_px = mapper.point(origin)
    angles = np.linspace(0.0, 2.0 * np.pi, env.cfg.n_rays, endpoint=False)
    dists = env._cast_rays()
    for angle, dist in zip(angles, dists):
        end = origin + np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(dist)
        pygame.draw.line(surface, (240, 220, 60), origin_px, mapper.point(end), 1)


def draw_scene(screen, font, env, mapper, episode_return, paused, label, last_action):
    screen.fill((18, 20, 24))
    boundary = pygame.Rect(mapper.margin, mapper.margin, mapper.plot_size, mapper.plot_size)
    pygame.draw.rect(screen, (230, 230, 230), boundary, 2)

    for ob in env.cfg.obstacles:
        rect = mapper.rect(ob)
        pygame.draw.rect(screen, (95, 100, 110), rect)
        pygame.draw.rect(screen, (180, 180, 190), rect, 2)

    if len(env.trajectory) > 1:
        pts = [mapper.point(p) for p in env.trajectory]
        pygame.draw.lines(screen, (245, 245, 245), False, pts, 2)

    draw_rays(screen, env, mapper)

    goal_px = mapper.point(env.goal)
    pygame.draw.circle(screen, (70, 220, 100), goal_px, max(5, mapper.length(env.cfg.goal_radius)))
    pygame.draw.circle(screen, (220, 255, 220), goal_px, max(5, mapper.length(env.cfg.goal_radius)), 2)

    drone_px = mapper.point(env.pos)
    pygame.draw.circle(screen, (80, 150, 255), drone_px, max(6, mapper.length(env.cfg.drone_radius) + 3))
    pygame.draw.circle(screen, (230, 245, 255), drone_px, max(6, mapper.length(env.cfg.drone_radius) + 3), 2)

    draw_arrow(screen, (80, 200, 255), drone_px, env.vel, scale=35, width=3)
    if last_action is not None:
        draw_arrow(screen, (255, 120, 80), drone_px, last_action, scale=30, width=2)

    dist_goal = float(np.linalg.norm(env.goal - env.pos))
    min_ray = float(np.min(env._cast_rays()))

    lines = [
        label,
        f"steps: {env.steps}/{env.cfg.max_steps}",
        f"return: {episode_return:+.1f}",
        f"distance to goal: {dist_goal:.2f}",
        f"min ray: {min_ray:.2f}",
        f"start/goal: {'manual' if env.cfg.fixed_start is not None and env.cfg.fixed_goal is not None else 'random/jitter'}",
        f"paused: {'yes' if paused else 'no'}",
        "R reset | N new random | SPACE pause | Q/ESC quit",
    ]
    y = 10
    for line in lines:
        img = font.render(line, True, (245, 245, 245))
        screen.blit(img, (10, y))
        y += 22


def main():
    parser = argparse.ArgumentParser(description="Pygame demo for custom PPO drone policy.")
    parser.add_argument("--difficulty", default="singlefixed")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default="best_model")
    parser.add_argument("--start", nargs=2, type=float, default=None)
    parser.add_argument("--goal", nargs=2, type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screen-size", type=int, default=850)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    agent, obs_normalizer = load_policy(args.model_dir, args.model_name, device=device)

    env = make_env(args.difficulty)
    base_env = env.unwrapped

    start = parse_pair(args.start)
    goal = parse_pair(args.goal)
    if start is not None:
        base_env.cfg.fixed_start = start
        base_env.cfg.jitter_start_center = None
    if goal is not None:
        base_env.cfg.fixed_goal = goal
        base_env.cfg.jitter_goal_center = None

    obs, info = env.reset(seed=args.seed)
    episode_return = 0.0
    done = False
    paused = False
    last_action = None
    final_status = ""

    pygame.init()
    pygame.display.set_caption("PPO Drone Navigation Demo")
    screen = pygame.display.set_mode((args.screen_size, args.screen_size))
    font = pygame.font.SysFont("consolas", 17)
    clock = pygame.time.Clock()
    mapper = Mapper(base_env.cfg.world_size, args.screen_size, args.margin)

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
                    last_action = None
                elif event.key == pygame.K_n:
                    obs, info = env.reset()
                    episode_return = 0.0
                    done = False
                    final_status = ""
                    last_action = None

        if not paused and not done:
            for _ in range(max(1, args.steps_per_frame)):
                obs_in = obs_normalizer.normalize(obs) if obs_normalizer is not None else obs
                action = agent.deterministic_action(obs_in)
                action = np.asarray(action, dtype=np.float32).reshape(2)
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

        label = f"PPO | {args.difficulty} | {args.model_name}"
        if final_status:
            label += f" | {final_status}"

        draw_scene(screen, font, base_env, mapper, episode_return, paused, label, last_action)
        pygame.display.flip()
        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

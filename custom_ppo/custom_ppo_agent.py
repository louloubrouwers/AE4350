"""
ppo.py
PPO agent 

This file takes one full RolloutBuffer, computes PPO's clipped policy loss + value loss + entropy bonus,
and updates the actor/critic networks with Adam.

Core PPO idea:
    ratio = pi_new(a|s) / pi_old(a|s)
    policy_loss = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))

The negative sign is because PyTorch optimizers minimize losses
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from custom_ppo.ppo_networks import GaussianActor, Critic
from custom_ppo.ppo_rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    # Optimization
    learning_rate: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 128
    max_grad_norm: float = 0.5

    # PPO objective
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: Optional[float] = None  

    # Advantage calculation
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True

    # Network
    hidden_size: int = 128


class PPOAgent:
    """Actor-critic PPO agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = config or PPOConfig()
        self.device = torch.device(device)

        hidden_sizes = (self.cfg.hidden_size, self.cfg.hidden_size)
        self.actor = GaussianActor(state_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.critic = Critic(state_dim, hidden_sizes=hidden_sizes).to(self.device)


        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.learning_rate,
            eps=1e-5,
        )

    # ------------------------------------------------------------------
    # Acting / value estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t = self.actor.act(state_t)
        value_t = self.critic(state_t)

        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        return action, log_prob, value

    @torch.no_grad()
    def deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """Use the Gaussian mean. This is for evaluation, not exploration/training."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(state_t)
        return dist.mean.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        """Return V(s) for a single numpy state."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(state_t).item())

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run several epochs of minibatch PPO updates over one completed rollout."""
        assert buffer.full, "RolloutBuffer must be full before calling update()"

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        total_losses = []

        early_stop = False

        for _epoch in range(self.cfg.update_epochs):
            for mb in buffer.get_minibatches(
                minibatch_size=self.cfg.minibatch_size,
                normalize_advantages=self.cfg.normalize_advantages,
            ):
                states = mb["states"]
                actions = mb["actions"]
                old_log_probs = mb["log_probs_old"]
                advantages = mb["advantages"]
                returns = mb["returns"]

                # Recompute log pi_new(a|s) under the current actor.
                new_log_probs, entropy = self.actor.evaluate(states, actions)
                values = self.critic(states)

                # Importance sampling ratio: pi_new / pi_old.
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_coef,
                    1.0 + self.cfg.clip_coef,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Critic regression loss: V(s) should match the lambda-return.
                value_loss = 0.5 * (returns - values).pow(2).mean()

                # Entropy is subtracted in the loss so larger entropy is rewarded.
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Common PPO diagnostic: approximate reverse KL(old || new).
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean().item()
                    )

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_mean.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                total_losses.append(loss.item())

                if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        with torch.no_grad():
            std_mean = self.actor.log_std.exp().mean().item()

        return {
            "loss": float(np.mean(total_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "std_mean": float(std_mean),
            "early_stop": float(early_stop),
        }

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.cfg),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "PPOAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = PPOConfig(**payload["config"])
        agent = cls(
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            config=cfg,
            device=device,
        )
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return agent


if __name__ == "__main__":
    # small test to verify that PPOAgent can be created and updated without errors
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim, action_dim = 14, 2
    cfg = PPOConfig(update_epochs=2, minibatch_size=8)
    agent = PPOAgent(state_dim, action_dim, cfg)
    buf = RolloutBuffer(
        rollout_steps=32,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    for _ in range(32):
        s = np.random.randn(state_dim).astype(np.float32)
        a, logp, v = agent.select_action(s)
        r = float(np.random.randn())
        buf.add(s, a, logp, r, v, done=False)
    buf.compute_returns_and_advantages(last_value=0.0, last_done=False)

    stats = agent.update(buf)
    print("PPO update OK:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

"""
rollout_buffer.py
Fixed-size on-policy rollout buffer for PPO with Generalized Advantage Estimation.

A "rollout" is a contiguous stretch of interaction with the environment under
the current policy. PPO collects `rollout_steps` transitions, computes advantages
and value targets, runs several gradient steps over the batch, and then discards
it (this is what makes PPO *on-policy*).

Stored per timestep:
    state       -- observation BEFORE the action was taken
    action      -- action taken
    log_prob    -- log pi_old(a|s), needed for the importance ratio later
    reward      -- env reward
    value       -- V_old(s), used in advantage calc
    done        -- True if episode terminated (NOT truncated) at this step

Computed at end of rollout:
    advantages  -- GAE-lambda advantages
    returns     -- value targets = advantages + values  (the "lambda-return")

Note on `done` vs `truncated`:
    In Gymnasium, terminated=True means a true terminal state (goal/collision
    here) -- future return is genuinely zero. truncated=True is just a time
    limit and the episode could have continued. PPO/GAE should bootstrap with
    V(s_next) on truncation but NOT on termination. We track this via `done`
    (terminated only) plus a final-value bootstrap parameter passed to
    `compute_returns_and_advantages`.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Pre-allocated numpy buffers for one PPO rollout."""

    def __init__(     #the training script decides how many steps to collect and what dimensions the state/action spaces have
        self,
        rollout_steps: int,    #how many transitions the buffer stores before PPO updates
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,   #how much you care about future reward. Here close to 1 so we care a lot 
        gae_lambda: float = 0.95,   #how much better or worse an action was than expected
        device: str = "cpu",
    ):
        self.rollout_steps = rollout_steps   
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # All buffers are numpy for fast assignment; we move to torch in `get`.
        self.states     = np.zeros((rollout_steps, state_dim), dtype=np.float32)   #each row is observation before taking action 
        self.actions    = np.zeros((rollout_steps, action_dim), dtype=np.float32)  #store the actions, each row is [ax,ay]
        self.log_probs  = np.zeros(rollout_steps, dtype=np.float32)   #store log π_old(a|s)
        self.rewards    = np.zeros(rollout_steps, dtype=np.float32)
        self.values     = np.zeros(rollout_steps, dtype=np.float32)
        self.dones      = np.zeros(rollout_steps, dtype=np.float32)  # terminated only

        # Filled by compute_returns_and_advantages
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)   #actual outcome - expected outcome. IF positive advantage = increase probability of that action
        self.returns    = np.zeros(rollout_steps, dtype=np.float32)

        self.ptr = 0
        self.full = False   #tells the rest of the code whether the buffer is ready for PPO update

    # -------------------- writing --------------------
    #store one transition
    #train_ppo.py calls: buffer.add(obs_n, action, log_prob, reward, value, done=terminated)
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        assert self.ptr < self.rollout_steps, "Buffer overflow -- call reset() first."
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = float(done)
        self.ptr += 1
        if self.ptr == self.rollout_steps:   #Now PPO can compute advantages and update the networks.
            self.full = True

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    # -------------------- GAE + returns --------------------

    def compute_returns_and_advantages(
        self,
        last_value: float,   #V(s_T), with s_T state after the final transition in the buffer
        last_done: bool,
    ) -> None:
        
        assert self.full, "Buffer not full; cannot compute advantages."

        adv = 0.0
        # next_value tracks V(s_{t+1}) at each iteration. Initially V(s_T),
        # which is `last_value` unless the final transition was terminal
        # (in which case s_T is the start of a new episode, not relevant).
        next_value = 0.0 if last_done else last_value

        for t in reversed(range(self.rollout_steps)):
            # non_terminal_t = 1 if the next state after step t is non-terminal,
            # else 0. This must use dones[t] *at iteration t*, not a value
            # carried over -- otherwise rewards bleed across episode boundaries
            # when an episode ends mid-buffer.
            non_terminal = 1.0 - self.dones[t]
            delta = (     #Positive delta means things went better than expected
                self.rewards[t]
                + self.gamma * next_value * non_terminal
                - self.values[t]
            )
            adv = delta + self.gamma * self.gae_lambda * non_terminal * adv
            self.advantages[t] = adv

            # For the NEXT iteration (t-1), V(s_t) is what we want as next_value.
            next_value = self.values[t]

        self.returns[:] = self.advantages + self.values

    # -------------------- reading --------------------

    def get_minibatches(
        self,
        minibatch_size: int,
        normalize_advantages: bool = True,
    ) -> Iterator[dict]:
        """
        Yield random minibatches over the full rollout for one epoch.

        Each minibatch is a dict of torch tensors (on self.device) with keys:
            states, actions, log_probs_old, advantages, returns, values_old
        """
        assert self.full, "Buffer not full; cannot iterate."

        # Advantage normalization is applied to a *copy* so it doesn't leak
        # across calls / epochs.
        advs = self.advantages.copy()
        if normalize_advantages:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        indices = np.random.permutation(self.rollout_steps)
        n_minibatches = self.rollout_steps // minibatch_size

        for mb in range(n_minibatches):
            idx = indices[mb * minibatch_size : (mb + 1) * minibatch_size]
            yield {
                "states":        torch.as_tensor(self.states[idx],    device=self.device),
                "actions":       torch.as_tensor(self.actions[idx],   device=self.device),
                "log_probs_old": torch.as_tensor(self.log_probs[idx], device=self.device),
                "advantages":    torch.as_tensor(advs[idx],           device=self.device),
                "returns":       torch.as_tensor(self.returns[idx],   device=self.device),
                "values_old":    torch.as_tensor(self.values[idx],    device=self.device),
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity-check: fill a small buffer with synthetic data and verify GAE math.
    buf = RolloutBuffer(rollout_steps=8, state_dim=4, action_dim=2,
                        gamma=0.99, gae_lambda=0.95)
    rng = np.random.default_rng(0)
    for t in range(8):
        buf.add(
            state=rng.normal(size=4).astype(np.float32),
            action=rng.uniform(-1, 1, size=2).astype(np.float32),
            log_prob=float(rng.normal()),
            reward=float(rng.normal()),
            value=float(rng.normal()),
            done=(t == 4),  # one terminal in the middle
        )
    buf.compute_returns_and_advantages(last_value=0.5, last_done=False)
    print("advantages:", buf.advantages)
    print("returns:   ", buf.returns)
    # Confirm: return_t = advantage_t + value_t (definition)
    assert np.allclose(buf.returns, buf.advantages + buf.values)
    print("\nGAE math consistent. Minibatching:")
    for i, mb in enumerate(buf.get_minibatches(minibatch_size=4)):
        print(f"  mb {i}: states {tuple(mb['states'].shape)}, "
              f"advs mean={mb['advantages'].mean().item():+.3f}, "
              f"std={mb['advantages'].std().item():.3f}")

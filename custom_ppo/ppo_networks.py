"""
Actor and critic networks used by the custom PPO agent.

The actor outputs a Gaussian policy for the continuous acceleration commands.
The critic estimates the scalar value function V(s). Both networks use two
hidden layers with tanh activations and orthogonal initialization.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2.0)) -> nn.Linear:
    """Orthogonal weight init + zero bias. Standard PPO init."""
    nn.init.orthogonal_(layer.weight, gain=gain) #initialize weight matrix of the layer using orthogonal initialization
    nn.init.zeros_(layer.bias) #set all biases to 0 to start
    return layer


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    # chosen network desing: compact 2-layer 
    # feedforward network with 128 neurons per layer.
    def __init__(
        self,
        state_dim: int,   #size of observation vector 
        action_dim: int,   #size of action vector
        hidden_sizes: Tuple[int, int] = (128, 128),
        log_std_init: float = -0.7,  #moderate exploration (not too random not too deterministic)
    ):
        super().__init__()  
        h1, h2 = hidden_sizes
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(state_dim, h1)),   #transform the observation vector into 128 hidden features
            nn.Tanh(),  # maps values into (-1,1) (learn non-linear control behaviour)
            _orthogonal_init(nn.Linear(h1, h2)),
            nn.Tanh(),
        )
        # Output layer uses a small gain (0.01) -- recommended in PPO papers
        # to keep initial actions near zero. => avoid huge acceleerations at start of training.
        self.mu_head = _orthogonal_init(nn.Linear(h2, action_dim), gain=0.01)  #maps the final hidden representation in mean action

        # State-independent learnable log_std (init exp(log_std)=1.0). RESULT ALWAYS POSITIVE
        # nn.Parameter means log_std is a learnable parameter of the model, not a fixed constant. => Pytorch will train it using gradient descent
        # PPO learns not only average action but also how much exploration to use 
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))


    # main function of the actor network: given a state, return the action distribution (a Normal distribution parameterized by mu and std)
    def forward(self, state: torch.Tensor) -> Normal:
        """Return the Normal distribution Normal(mu(s), exp(log_std))."""
        h = self.trunk(state)   #send input state through the hidden layers
        mu = self.mu_head(h)   #compute mean action
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)  #return pytorch prob distribution 
    
    #important: mean depends on the state, but the standard deviation does not depend on the state.

    @torch.no_grad()   #Do not track gradients inside this function.
    # used when the drone is interacting with the env and collecitng experience
    #during rollout we are not training yet, we are using the ucrretn policy
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(state)  #create normal Gauss distribution for the given state
        action = dist.sample()  #sample an action from the distribution
        # Sum log-probs over action dimensions: log pi(a|s) = sum_i log pi(a_i|s)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


    #used during training/update, with gradients
    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        dist = self.forward(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Sum entropy over action dimensions
        entropy = dist.entropy().sum(dim=-1)   #measures how random the policy is
        return log_prob, entropy  #retun the two actor quantitities needed during PPO training


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """Scalar state-value network V(s)."""  #expected future discounted reward starting from state s

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, int] = (128, 128),
    ):
        super().__init__()   #inti parent PyTorch class nn.Module
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(state_dim, h1)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            # Value head uses gain=1.0 (typical for regression outputs) => does not need to be near zero as tightly as the actor mean
            _orthogonal_init(nn.Linear(h2, 1), gain=1.0),
        )

    # compute the value estimate for a given state
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)   #ex: (batch_size, 1) => (batch_size,)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)    # fix random seed for reproducibility
    state_dim, action_dim, batch = 14, 2, 8   #fake setup 

    actor = GaussianActor(state_dim, action_dim)
    critic = Critic(state_dim)
    print(actor)    #print network architectures
    print(critic)
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters())}")

    s = torch.randn(batch, state_dim)  #create random fake states

    # Rollout-time: act
    a, logp = actor.act(s)
    print(f"action shape={tuple(a.shape)}, log_prob shape={tuple(logp.shape)}")

    # Update-time: evaluate (with grad)
    logp2, ent = actor.evaluate(s, a)
    v = critic(s)
    print(f"evaluate log_prob shape={tuple(logp2.shape)}, entropy shape={tuple(ent.shape)}")
    print(f"value shape={tuple(v.shape)}")

    # Check backprop works through actor and critic
    loss = -logp2.mean() - 0.01 * ent.mean() + (v ** 2).mean()  #figuring out which knobs caused the badness
    loss.backward()
    print("Backprop OK.")

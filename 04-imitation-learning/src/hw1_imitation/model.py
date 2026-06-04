"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # ╔═══════════════════ STEP 1 — DONE ════════════════════╗
        # MLP: state_dim -> hidden_dims (Linear+ReLU each) -> chunk_size*action_dim.
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))  # no ReLU on output
        self.net = nn.Sequential(*layers)
        # ╚═══════════════════ STEP 1 — END ═════════════════════╝

    def compute_loss(
        self,
        state: torch.Tensor,          # (B, state_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size, action_dim)  <- expert target
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 2 — START ═══════════════════╗
        # MSE loss (hw1 eq. 1):
        #   run self.net on `state`, reshape output to match action_chunk,
        #   return the mean squared error vs the expert chunk.
        prediction = self.net(state)  # line 1: (B, 5) -> (B, 16) flat
        prediction = prediction.reshape(-1, self.chunk_size, self.action_dim)  # line 2: (B, 8, 2)
        return nn.functional.mse_loss(prediction, action_chunk)  # line 3: mean squared error (eq. 1)
        # ╚═══════════════════ STEP 2 — END ═════════════════════╝

    def sample_actions(
        self,
        state: torch.Tensor,          # (B, state_dim)
        *,
        num_steps: int = 10,          # ignored for MSE (only flow uses it)
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 3 — START ═══════════════════╗
        # Produce a chunk:
        #   run self.net on `state`, reshape to (B, chunk_size, action_dim), return it.
        prediction = self.net(state)  # line 1: (B, 5) -> (B, 16) flat
        return prediction.reshape(-1, self.chunk_size, self.action_dim)  # line 2: (B, 8, 2) chunk
        # ╚═══════════════════ STEP 3 — END ═════════════════════╝
        # ----- after STEP 3: go do STEP 4 = the training loop in train.py, then run MSE -----


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # ╔═══════════════════ STEP 5 — DONE ════════════════════╗
        # WHAT: build the velocity-field network v_theta(o, A_tau, tau).
        # WHY : flow predicts how to MOVE noise toward a real chunk — so its input
        #       also carries the noisy chunk and the timestep tau, not just the state.
        layers: list[nn.Module] = []
        in_dim = state_dim + chunk_size * action_dim + 1  # o(5) + A_tau(16) + tau(1) = 22
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))  # -> velocity, 16-D (no ReLU)
        self.net = nn.Sequential(*layers)
        # ╚═══════════════════ STEP 5 — END ═════════════════════╝

    def compute_loss(
        self,
        state: torch.Tensor,          # (B, state_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size, action_dim)  <- clean chunk A_t
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 6 — DONE ════════════════════╗
        # WHAT: the flow-matching loss (hw1 eq. 2).
        # WHY : teach v_theta the straight-line velocity from noise to the real chunk.
        # >>> write STEP 6 here, then delete the raise below <<<
        noise = torch.randn_like(action_chunk)  # A_{t,0} ~ N(0, I), shape (B, 8, 2)
        batch_size = action_chunk.shape[0]
        tau = torch.rand(batch_size, 1, device=action_chunk.device)  # tau ~ U(0,1), (B, 1)
        tau_chunk = tau.unsqueeze(-1)  # (B, 1, 1) so it broadcasts over the 8x2 chunk
        noisy_chunk = tau_chunk * action_chunk + (1.0 - tau_chunk) * noise  # A_{t,tau}, (B, 8, 2)
        net_input = torch.cat([state, noisy_chunk.flatten(1), tau], dim=1)  # (B, 5+16+1)=(B, 22)
        velocity = self.net(net_input).reshape(batch_size, self.chunk_size, self.action_dim)  # (B,8,2)
        target = action_chunk - noise  # A_t - A_{t,0}, straight-line velocity (data - noise), (B,8,2)
        return nn.functional.mse_loss(velocity, target)  # eq. 2: mean ||v_theta - (A_t - A_0)||^2
        # ╚═══════════════════ STEP 6 — END ═════════════════════╝

    def sample_actions(
        self,
        state: torch.Tensor,          # (B, state_dim)
        *,
        num_steps: int = 10,          # number of Euler integration steps n
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 7 — DONE ════════════════════╗
        # WHAT: generate a chunk by integrating the ODE (hw1 eq. 3).
        # WHY : at inference there's no data — start from noise, follow v_theta to a chunk.
        # >>> write STEP 7 here, then delete the raise below <<<
        batch_size = state.shape[0]
        chunk = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=state.device
        )  # A_{t,0} ~ N(0, I), shape (B, 8, 2) — start of the walk
        dt = 1.0 / num_steps  # Euler step size; n steps carry tau from 0 to 1
        for step in range(num_steps):
            tau = torch.full((batch_size, 1), step * dt, device=state.device)  # τ=i/n, (B,1)
            net_input = torch.cat([state, chunk.flatten(1), tau], dim=1)  # (B, 22)
            velocity = self.net(net_input).reshape(
                batch_size, self.chunk_size, self.action_dim
            )  # v_theta at current point, (B, 8, 2)
            chunk = chunk + dt * velocity  # Euler step (eq. 3): advance toward data
        return chunk  # A_{t,1}: noise has been integrated into a real action chunk, (B, 8, 2)
        # ╚═══════════════════ STEP 7 — END ═════════════════════╝


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

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
        # ╔═══════════════════ STEP 5 — START ═══════════════════╗  (after MSE works)
        # WHAT: build the velocity-field network v_theta(o, A_tau, tau).
        # WHY : flow predicts how to MOVE noise toward a real chunk — so its input
        #       also carries the noisy chunk and the timestep tau, not just the state.
        # >>> write STEP 5 here <<<
        # ╚═══════════════════ STEP 5 — END ═════════════════════╝

    def compute_loss(
        self,
        state: torch.Tensor,          # (B, state_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size, action_dim)  <- clean chunk A_t
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 6 — START ═══════════════════╗
        # WHAT: the flow-matching loss (hw1 eq. 2).
        # WHY : teach v_theta the straight-line velocity from noise to the real chunk.
        # >>> write STEP 6 here, then delete the raise below <<<
        raise NotImplementedError
        # ╚═══════════════════ STEP 6 — END ═════════════════════╝

    def sample_actions(
        self,
        state: torch.Tensor,          # (B, state_dim)
        *,
        num_steps: int = 10,          # number of Euler integration steps n
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 7 — START ═══════════════════╗
        # WHAT: generate a chunk by integrating the ODE (hw1 eq. 3).
        # WHY : at inference there's no data — start from noise, follow v_theta to a chunk.
        # >>> write STEP 7 here, then delete the raise below <<<
        raise NotImplementedError
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

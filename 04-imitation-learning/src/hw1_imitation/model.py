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
        # ╔═══════════════════ STEP 1 — START ═══════════════════╗
        # Build the MLP.
        #   input  = state_dim (= 5)
        #   output = chunk_size * action_dim (= 8 * 2 = 16)
        #   Linear -> ReLU across hidden_dims; NO activation on the last layer.
        #   Save it as self.net so STEP 2 and STEP 3 can call it.
        #
        # >>> write STEP 1 here <<<
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
        #
        # >>> write STEP 2 here, then delete the raise below <<<
        raise NotImplementedError
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
        #
        # >>> write STEP 3 here, then delete the raise below <<<
        raise NotImplementedError
        # ╚═══════════════════ STEP 3 — END ═════════════════════╝
        # ----- after STEP 3: go write the training loop in train.py:130, then run MSE -----


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
        # ╔═══════════════════ STEP 4 — START ═══════════════════╗  (do this AFTER MSE works)
        # Build the velocity-field MLP v_theta(o, A_tau, tau).
        #   input  = state_dim + chunk_size*action_dim + 1   (state + noisy chunk + tau)
        #   output = chunk_size*action_dim                   (the velocity)
        #   Save it as self.net.
        #
        # >>> write STEP 4 here <<<
        # ╚═══════════════════ STEP 4 — END ═════════════════════╝

    def compute_loss(
        self,
        state: torch.Tensor,          # (B, state_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size, action_dim)  <- clean chunk A_t
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 5 — START ═══════════════════╗
        # Flow-matching loss (hw1 eq. 2):
        #   sample A_0 ~ N(0,I) and tau ~ U(0,1)
        #   interpolate  A_tau = tau*A_t + (1-tau)*A_0
        #   predict velocity, regress onto target (A_t - A_0), return MSE.
        #
        # >>> write STEP 5 here, then delete the raise below <<<
        raise NotImplementedError
        # ╚═══════════════════ STEP 5 — END ═════════════════════╝

    def sample_actions(
        self,
        state: torch.Tensor,          # (B, state_dim)
        *,
        num_steps: int = 10,          # number of Euler integration steps n
    ) -> torch.Tensor:
        # ╔═══════════════════ STEP 6 — START ═══════════════════╗
        # Integrate the ODE (hw1 eq. 3):
        #   start from A_0 ~ N(0,I); Euler-step tau: 0 -> 1 in num_steps steps;
        #   return the final chunk (B, chunk_size, action_dim).
        #
        # >>> write STEP 6 here, then delete the raise below <<<
        raise NotImplementedError
        # ╚═══════════════════ STEP 6 — END ═════════════════════╝


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

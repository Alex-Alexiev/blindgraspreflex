"""RNN actor whose trailing output dims are an observation-independent learnable constant.

Motivation: the grasp policy's action is ``[grasp_score(1), arm_residual(6), grasp_pose(H),
reach_pose(H)]``. The REACH pose is the hand shape held while the arm approaches the object
(before engagement). We want that reach pose to be a *single tuned constant* -- the same 20-DOF
(or 5-D PCA) pose in every state -- rather than a function of the observation.

Implementation: keep the reach dims as real action dimensions so PPO still optimizes them by
policy gradient (they are sampled from the Gaussian and get advantage-weighted updates), but make
their *mean* an ``nn.Parameter`` bias spliced in after the MLP instead of a function of the RNN
latent. The mean is therefore obs-independent; the shared action std still provides exploration
and shrinks as the constant converges. Only the trailing ``REFLEXIVE_CONST_REACH_DIM`` output dims
are replaced; the leading dims (grasp score, arm residual, grasp pose) stay obs-conditioned.

The constant count is read from the ``REFLEXIVE_CONST_REACH_DIM`` env var (set by train_grasp) so
no new field has to be threaded through the frozen RSL-RL model config. 0/unset => behaves exactly
like the base RNNModel (safe no-op).

NOTE: the JIT/ONNX export helpers inherited from RNNModel copy only ``self.mlp`` and do NOT splice
the constant, so an *exported* model would be wrong. Export isn't used for training, the viewer, or
eval (all go through ``forward``); if this teacher is later exported/distilled, override ``as_jit``/
``as_onnx`` too.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.utils import unpad_trajectories

_ENV_VAR = "REFLEXIVE_CONST_REACH_DIM"


class RNNModelConstReach(RNNModel):
    """RNNModel whose last ``REFLEXIVE_CONST_REACH_DIM`` mean-outputs are a learnable constant."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._const_dim = int(os.environ.get(_ENV_VAR, "0"))
        # Bias initialized to zeros: with the sigmoid squash in the action term this maps to the
        # mid-range joint pose (sigmoid(0)=0.5), matching the near-zero init of the network head it
        # replaces, so training starts from the same reach pose either way.
        self.reach_bias = (
            nn.Parameter(torch.zeros(self._const_dim)) if self._const_dim > 0 else None
        )

    def _splice_const(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Replace the trailing ``_const_dim`` columns of the mean with the learnable constant."""
        if not self._const_dim:
            return mlp_output
        head = mlp_output[..., : mlp_output.shape[-1] - self._const_dim]
        const = self.reach_bias.to(mlp_output.dtype)
        const = const.expand(*mlp_output.shape[:-1], self._const_dim)
        return torch.cat([head, const], dim=-1)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        # Mirror MLPModel.forward, splicing the constant reach mean in before the distribution so
        # sampling, deterministic inference, and the PPO update's log-prob/KL all see the constant.
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self._splice_const(self.mlp(latent))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

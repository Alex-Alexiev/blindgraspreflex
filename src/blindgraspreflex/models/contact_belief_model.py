"""Contact-belief distillation: a recurrent student that reconstructs the teacher's privileged 16
hand contacts from a short proprioception history (supervised BCE), and acts on that contact-aware
latent — dropping palm_to_object entirely.

Rationale. The teacher sees privileged 16-link contacts + palm_to_object. On hardware neither is
measurable, but proprioception IS: a finger blocked by contact lags its commanded target, so a short
history of ``[joint_pos, target-joint_pos]`` encodes contact. We give the student that history, add a
head that regresses the 16 contacts (BCE against the teacher's ground truth), and let the policy act
on the shared recurrent latent that the contact loss shapes. This is the standard belief/RMA-style
privileged-information distillation: the auxiliary contact task forces the latent to encode contact,
which the policy then uses in place of the dropped palm vector.

Wiring (see distill_grasp.build_env(belief_student=True) / build_cfg):
  student obs group  = proprio history   (hand_joint_pos + hand_tracking, history_length=N)  -> RNN latent
  contacts obs group = contact_binary(16)  (NOT listed in obs_groups; the BCE target only)
  teacher obs group  = 16 contacts + palm_to_object + proprio  (unchanged, so teacher weights load)

The auxiliary weight is read from env REFLEXIVE_CONTACT_WEIGHT (default 1.0).
"""

from __future__ import annotations

import os

import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories


class ContactBeliefModel(RNNModel):
    """RNNModel + an auxiliary 16-contact head off the shared recurrent latent.

    ``forward`` still returns ONLY the action tensor (so ``act``/inference/export are unchanged); the
    16 contact LOGITS are cached on ``self.last_contact_logits`` for the algorithm's BCE loss. The
    head reads the same latent the policy does, produced by the single ``get_latent`` call — the RNN
    hidden state must advance exactly once per step, so contacts are NOT computed in a second forward.
    """

    def __init__(self, *args, num_contacts: int = 16, contact_hidden_dims=(128,), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_contacts = int(num_contacts)
        # logits head (BCE-with-logits downstream), off the recurrent latent (dim = _get_latent_dim)
        self.contact_head = MLP(
            self._get_latent_dim(), self.num_contacts, list(contact_hidden_dims), "elu"
        )
        self.last_contact_logits = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)  # advances the RNN once
        self.last_contact_logits = self.contact_head(latent)  # cached side-output for the aux loss
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output


class ContactDistillation(Distillation):
    """Distillation + a supervised contact-BCE auxiliary loss.

    Reads the ground-truth 16 contacts from the ``contacts`` obs group (present in every stored
    transition, but never read as a model input) and adds
    ``contact_weight * BCE_with_logits(student.last_contact_logits, contacts)`` into the SAME
    truncated-BPTT accumulation window as the behaviour-cloning loss. Body copied from
    ``rsl_rl.algorithms.Distillation.update`` with the two contact lines added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_weight = float(os.environ.get("REFLEXIVE_CONTACT_WEIGHT", "1.0"))

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_contact_loss = 0
        loss = 0
        cnt = 0

        for _epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                actions = self.student(batch.observations)  # also caches last_contact_logits
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)
                true_contacts = batch.observations["contacts"].float()
                contact_loss = F.binary_cross_entropy_with_logits(
                    self.student.last_contact_logits, true_contacts
                )

                loss = loss + behavior_loss + self.contact_weight * contact_loss
                mean_behavior_loss += behavior_loss.item()
                mean_contact_loss += contact_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        mean_contact_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()
        return {"behavior": mean_behavior_loss, "contact": mean_contact_loss}

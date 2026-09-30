import numpy as np
import torch

from blindgraspreflex.embodiments.pca import HandPCA
from blindgraspreflex.method import HandPCAProjector, QCommandHistory


def test_q_command_history_is_oldest_to_newest() -> None:
    history = QCommandHistory(num_envs=1, hand_dof=2, history_length=2)
    q0 = torch.tensor([[1.0, 2.0]])
    history.reset(q0, q0 + 0.5)
    q1 = torch.tensor([[3.0, 4.0]])
    result = history.update(q1, q1 + 1.0)
    assert torch.equal(result, torch.tensor([[1.0, 2.0, 0.5, 0.5, 3.0, 4.0, 1.0, 1.0]]))


def test_half_projection_blends_original_and_manifold() -> None:
    pca = HandPCA(
        components=np.array([[1.0, 0.0]], dtype=np.float32),
        mean=np.zeros(2, dtype=np.float32),
        lower=np.array([-1.0], dtype=np.float32),
        upper=np.array([1.0], dtype=np.float32),
    )
    projector = HandPCAProjector(pca, alpha=0.5)
    result = projector(torch.tensor([[2.0, 4.0]]))
    assert torch.equal(result, torch.tensor([[2.0, 2.0]]))


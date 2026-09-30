from blindgraspreflex.config import CANONICAL_METHOD


def test_q_cmd_observation_size_tracks_hand_dof() -> None:
    assert CANONICAL_METHOD.student_observation_size(20) == 200
    assert CANONICAL_METHOD.student_observation_size(12) == 120


def test_canonical_method_is_q_cmd() -> None:
    assert CANONICAL_METHOD.student.observation == "q_cmd"
    assert CANONICAL_METHOD.student.history_length == 5
    assert CANONICAL_METHOD.hand_pca_projection == 0.5

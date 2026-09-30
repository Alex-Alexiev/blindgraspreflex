from blindgraspreflex.embodiments.pca import load_hand_pca
from blindgraspreflex.embodiments.robotis_5f import make_embodiment


def test_robotis_metadata_and_pca_are_self_consistent() -> None:
    embodiment = make_embodiment()
    assert embodiment.validate(require_mjcf=False) == []
    pca = load_hand_pca(embodiment.hand.pca_directory, embodiment.hand.joint_names)
    assert pca.components.shape == (5, 20)
    assert pca.mean.shape == (20,)


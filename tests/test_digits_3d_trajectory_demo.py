from pathlib import Path


def test_demo_saves_a_resumable_checkpoint_at_every_phase_boundary():
    script = (Path(__file__).parents[1] / "demos/digits_3d_trajectory_demo.sh").read_text(
        encoding="utf-8"
    )

    assert "--only_save_checkpoint_at_end" in script

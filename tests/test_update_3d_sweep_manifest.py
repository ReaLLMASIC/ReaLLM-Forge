import json
import runpy
from pathlib import Path


MODULE = runpy.run_path(Path(__file__).parents[1] / "analysis/update_3d_sweep_manifest.py")
update_manifest = MODULE["update_manifest"]


def write_run(path, embedding_dim, trained, held_out, fixed_norm=None, weight_tying=True, dropped=1):
    path.write_text(json.dumps({
        "projection": {"method": "pca", "input_dimensions": embedding_dim},
        "trained_tokens": list(range(trained)),
        "unseen_tokens": list(range(held_out)),
        "fixed_norm": fixed_norm,
        "wte_weight_tying": weight_tying,
        "dropped_tokens": list(range(dropped)),
        "dropout_iteration": 500,
        "frames": [{"iteration": 1000}],
        "transition_mode": "drop",
        "transition_iterations": [500],
        "optimizer": {"name": "adam", "weight_decay": 0.1, "muon_include_all_weights": False},
    }), encoding="utf-8")


def test_manifest_is_updated_with_every_completed_run(tmp_path):
    write_run(tmp_path / "dim-8_first.json", 8, 5, 2)
    destination = update_manifest(tmp_path)
    first = json.loads(destination.read_text(encoding="utf-8"))
    assert [run["name"] for run in first["runs"]] == ["dim-8_first"]

    write_run(tmp_path / "dim-16_second.json", 16, 10, 4, fixed_norm=4.0, weight_tying=False)
    update_manifest(tmp_path)
    second = json.loads(destination.read_text(encoding="utf-8"))
    assert [run["name"] for run in second["runs"]] == ["dim-16_second", "dim-8_first"]
    assert second["runs"][0]["fixed_norm"] == 4.0
    assert second["runs"][0]["wte_weight_tying"] is False
    assert second["runs"][0]["dropped_tokens"] == 1
    assert second["runs"][0]["dropout_iteration"] == 500
    assert second["runs"][0]["final_iteration"] == 1000
    assert second["runs"][0]["transition_mode"] == "drop"
    assert second["runs"][0]["transition_iterations"] == [500]
    assert second["runs"][0]["optimizer"]["name"] == "adam"
    assert second["runs"][0]["optimizer"]["weight_decay"] == 0.1
    assert not (tmp_path / ".manifest.json.tmp").exists()

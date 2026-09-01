import json
import runpy
from pathlib import Path

import pytest


MODULE = runpy.run_path(Path(__file__).parents[1] / "analysis/package_3d_trajectory_site.py")
package_site = MODULE["package_site"]


def make_source(root):
    (root / "runs").mkdir(parents=True)
    (root / "index.html").write_text("selector using viewer.html?data=", encoding="utf-8")
    (root / "viewer.html").write_text("viewer", encoding="utf-8")


def test_packages_completed_runs_as_site_root(tmp_path):
    source, output = tmp_path / "source", tmp_path / "site"
    make_source(source)
    (source / "runs/dim-3_example.json").write_text(json.dumps({
        "projection": {"method": "native", "input_dimensions": 3},
        "trained_tokens": ["0"], "unseen_tokens": ["a"],
        "frames": [{"iteration": 10}],
    }), encoding="utf-8")

    assert package_site(source, output) == 1
    assert (output / ".nojekyll").exists()
    assert (output / "viewer.html").read_text(encoding="utf-8") == "viewer"
    assert "viewer.html?data=" in (output / "index.html").read_text(encoding="utf-8")
    assert (output / "runs/dim-3_example.json").exists()
    assert len(json.loads((output / "runs/manifest.json").read_text())["runs"]) == 1


def test_refuses_to_package_empty_site(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    with pytest.raises(ValueError, match="no completed trajectory"):
        package_site(source, tmp_path / "site")

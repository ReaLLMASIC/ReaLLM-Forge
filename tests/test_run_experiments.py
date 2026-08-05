import importlib.util
from pathlib import Path
import unittest

import yaml


MODULE_PATH = (
    Path(__file__).parents[1] / "optimization_and_search" / "run_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("run_experiments", MODULE_PATH)
run_experiments = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_experiments)


class SweepDefaultAndNullTests(unittest.TestCase):
    def test_unquoted_default_omits_parameter_and_null_preserves_none(self):
        config = yaml.safe_load(
            'norm_variant_wte: [default, "hyperspherenorm", null]'
        )

        combinations = [
            combo for combo, _common_keys in run_experiments.generate_combinations(config)
        ]

        self.assertEqual(
            combinations,
            [
                {},
                {"norm_variant_wte": "hyperspherenorm"},
                {"norm_variant_wte": None},
            ],
        )

    def test_null_is_not_rendered_as_a_cli_string(self):
        command = run_experiments.build_command(
            {
                "norm_variant_wte": None,
                "activation_variant": "gelu",
            }
        )

        self.assertEqual(command, ["python3", "train.py", "--activation_variant", "gelu"])

    def test_default_in_common_group_is_omitted(self):
        config = {"common_group": {"norm_variant_wte": ["default"]}}

        self.assertEqual(list(run_experiments.generate_combinations(config)), [({}, set())])


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from run_exploration_monitor import (
    ExplorationConfigScreen,
    MonitorApp,
    load_exploration_fields,
)


class ColumnSettingRemapTests(unittest.TestCase):
    def setUp(self):
        self.app = object.__new__(MonitorApp)
        self.app.current_entries = [{"short_column_name": "abc"}]
        self.app.auto_fit_columns = set()
        self.app.tight_fit_columns = set()

    def test_width_cycle_adds_tight_fit_smaller_than_column_heading(self):
        column = "short_column_name"

        self.assertEqual(self.app._column_width(column), len(column) + 2)
        self.assertEqual(self.app._cycle_column_width(column), "fit to data")
        self.assertEqual(self.app._column_width(column), len(column) + 2)
        self.assertEqual(self.app._cycle_column_width(column), "tightly fit to data")
        self.assertEqual(self.app._column_width(column), len("abc") + 2)
        self.assertEqual(self.app._cycle_column_width(column), "reset")
        self.assertEqual(self.app._column_width(column), len(column) + 2)

    def test_tight_fit_still_uses_largest_visible_value(self):
        column = "value"
        self.app.current_entries = [{column: "a"}, {column: "longest"}]
        self.app.tight_fit_columns.add(column)

        self.assertEqual(self.app._column_width(column), len("longest") + 2)

    def test_title_and_exploration_config_use_log_yaml_name(self):
        app = MonitorApp(
            log_file=Path("exploration_logs/default.yaml"),
            interval=30.0,
            csv_dir="rem_csv_exports",
        )

        self.assertEqual(app.title, "default.yaml")
        self.assertEqual(app.sub_title, "exploration_logs/default.yaml")
        self.assertEqual(app.exploration_config_file.name, "default.yaml")
        self.assertEqual(app.exploration_config_file.parent.name, "explorations")

    def test_colour_and_sort_settings_follow_reordered_columns(self):
        app = object.__new__(MonitorApp)
        app.columns = ["gamma", "alpha", "beta"]
        app.colour_columns = {0: "high_low", 2: "low_high"}
        app.sort_stack = [(1, True), (0, False)]

        app._remap_indexed_column_settings(["alpha", "beta", "gamma"])

        self.assertEqual(app.colour_columns, {1: "high_low", 0: "low_high"})
        self.assertEqual(app.sort_stack, [(2, True), (1, False)])


class ExplorationConfigScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_hotkey_opens_associated_yaml_contents(self):
        app = MonitorApp(
            log_file=Path("exploration_logs/default.yaml"),
            interval=3600.0,
            csv_dir="rem_csv_exports",
        )

        async with app.run_test() as pilot:
            await pilot.press("I")

            self.assertIsInstance(app.screen, ExplorationConfigScreen)
            contents = app.screen.query_one("#config-contents").content
            self.assertIn("norm_variant_wte", str(contents))


class ExplorationFieldRefreshTests(unittest.IsolatedAsyncioTestCase):
    def test_loads_flat_and_grouped_fields_but_not_schema_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "sweep.yaml"
            config_file.write_text(
                """\
max_iters: [100]
named_static_groups:
  - named_group: defaults
    named_group_settings:
      use_qk_norm: [true]
parameter_groups:
  - learning_rate: [0.001]
    named_group_static: [defaults]
"""
            )

            self.assertEqual(
                load_exploration_fields(config_file),
                {"max_iters", "use_qk_norm", "learning_rate"},
            )

    async def test_refresh_and_hotkey_add_latest_exploration_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_file = root / "logs" / "sweep.yaml"
            log_file.parent.mkdir()
            log_file.write_text("config:\n  max_iters: 100\n")
            config_file = root / "explorations" / "sweep.yaml"
            config_file.parent.mkdir()
            config_file.write_text("max_iters: [100]\n")

            app = MonitorApp(log_file, interval=3600.0, csv_dir=tmpdir)
            app.exploration_config_file = config_file
            async with app.run_test() as pilot:
                self.assertNotIn("learning_rate", app.columns)

                config_file.write_text(
                    "max_iters: [100]\nlearning_rate: [0.001]\n"
                )
                app.refresh_table()
                self.assertIn("learning_rate", app.columns)

                config_file.write_text(
                    "max_iters: [100]\nlearning_rate: [0.001]\ndropout: [0.1]\n"
                )
                await pilot.press("R")
                self.assertIn("dropout", app.columns)


if __name__ == "__main__":
    unittest.main()

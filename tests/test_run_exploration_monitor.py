import unittest
from pathlib import Path

from run_exploration_monitor import ExplorationConfigScreen, MonitorApp


class ColumnSettingRemapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

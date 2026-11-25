import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import aracne
from lib.input_history import GoalHistory


class FakeReadline:
    def __init__(self):
        self.entries = []
        self.auto_history = True

    def clear_history(self):
        self.entries.clear()

    def read_history_file(self, path):
        self.entries.extend(Path(path).read_text().splitlines())

    def set_history_length(self, length):
        self.limit = length

    def set_auto_history(self, enabled):
        self.auto_history = enabled

    def get_current_history_length(self):
        return len(self.entries)

    def get_history_item(self, index):
        return self.entries[index - 1]

    def add_history(self, entry):
        self.entries.append(entry)

    def write_history_file(self, path):
        Path(path).write_text("\n".join(self.entries[-self.limit:]) + "\n")


class InputHistoryTests(unittest.TestCase):
    def test_goals_are_saved_private_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input_history"
            readline = FakeReadline()
            with patch("sys.stdin.isatty", return_value=True), patch.dict(sys.modules, {"readline": readline}):
                history = GoalHistory(path)
                self.assertTrue(history.enable())
                history.add("find hosts")
                history.add("find hosts")
                history.add("inspect 192.0.2.3")
                history.add("quit")
            self.assertEqual(readline.entries, ["find hosts", "inspect 192.0.2.3"])
            self.assertFalse(readline.auto_history)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            next_readline = FakeReadline()
            with patch("sys.stdin.isatty", return_value=True), patch.dict(sys.modules, {"readline": next_readline}):
                self.assertTrue(GoalHistory(path).enable())
            self.assertEqual(next_readline.entries, readline.entries)

    def test_goal_prompt_records_only_accepted_goals(self):
        with patch("builtins.input", side_effect=["", "inspect host"]), patch.object(
            GoalHistory, "add"
        ) as add:
            self.assertEqual(aracne.prompt_new_goal(GoalHistory("unused")), "inspect host")
        add.assert_called_once_with("inspect host")


if __name__ == "__main__":
    unittest.main()

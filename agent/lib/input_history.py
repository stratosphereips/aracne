"""Readline history for goals entered at ARACNE's terminal prompts."""

import os
import sys
from pathlib import Path


class GoalHistory:
    MAX_ENTRIES = 500

    def __init__(self, path):
        self.path = Path(path)
        self.readline = None

    def enable(self):
        """Enable arrow-key editing when stdin is an interactive terminal."""
        if not sys.stdin.isatty():
            return False
        try:
            import readline
        except ImportError:
            return False

        self.readline = readline
        readline.clear_history()
        try:
            readline.read_history_file(str(self.path))
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            # A bad history file must not prevent an interactive session.
            pass
        readline.set_history_length(self.MAX_ENTRIES)
        if hasattr(readline, "set_auto_history"):
            readline.set_auto_history(False)
        return True

    def add(self, goal):
        """Make an accepted goal available immediately and on the next run."""
        if self.readline is None:
            return
        goal = str(goal).strip()
        if not goal or goal.lower() in {"quit", "exit"}:
            return
        count = self.readline.get_current_history_length()
        if count and self.readline.get_history_item(count) == goal:
            return
        self.readline.add_history(goal)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600)
            os.close(descriptor)
            self.path.chmod(0o600)
            self.readline.write_history_file(str(self.path))
        except OSError:
            # In-memory history still works if the file cannot be written.
            pass

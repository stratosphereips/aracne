import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from lib import planner
from lib.utils import load_prompt
from lib.memory import InteractiveMemory


class InteractiveMemoryTests(unittest.TestCase):
    def test_large_output_is_condensed_and_context_stays_bounded(self):
        memory = InteractiveMemory()
        with patch("lib.memory.call_model", return_value="Found host 192.168.0.7") as summarize:
            memory.record_action("find hosts", "nmap -sn 192.168.0.0/24", {
                "output": "scan output\n" * 2000,
                "exit_status": 0,
                "timed_out": False,
            })
            memory.start_goal("inspect the found host")
            for index in range(12):
                memory.record_action("inspect the found host", f"command {index}", {
                    "output": "x" * 1000,
                    "exit_status": 0,
                    "timed_out": False,
                })
        self.assertTrue(summarize.called)
        self.assertLessEqual(len(memory.context()), memory.MAX_CONTEXT_CHARS)
        self.assertIn("192.168.0.7", memory.context())
        self.assertIn("command 11", memory.context())

    def test_new_goal_prompt_contains_prior_context(self):
        prior = "Found host 192.168.0.7 in previous goal"
        response = '{"steps": ["Inspect 192.168.0.7"], "goal_reached": false, "goal_verification": "The host has not been inspected yet."}'
        with patch.object(planner, "call_model", return_value=response) as call:
            planner.generate_plan("Inspect the found host", prior_context=prior)
        self.assertIn(prior, call.call_args.args[0])
        self.assertIn("judge goal_reached only for the current goal", call.call_args.args[0])

    def test_planner_prompt_renders_valid_json_example(self):
        prompt = planner.PROMPT.format(goal="Find the current user", summary="")
        example = next(line.split("Example: ", 1)[1] for line in prompt.splitlines() if "Example: " in line)
        parsed = json.loads(example)
        self.assertEqual(parsed["goal_reached"], False)
        self.assertIn("goal_verification", parsed)
        self.assertNotIn("dummy ssh", prompt.lower())

    def test_interpreter_and_summarizer_prompts_render_with_shell_evidence(self):
        command = "printf '%s' '{\"host\":\"192.0.2.3\"}'"
        interpreter_prompt = load_prompt("config/interpreterConfig.yml").format(
            instruction=f"Run {command} on the configured host"
        )
        summary_prompt = load_prompt("config/summarizerConfig.yml").format(
            previous_summary="Found 192.0.2.3:22",
            new_context=f"COMMAND: {command}\nEXIT STATUS: 0",
        )
        self.assertIn(command, interpreter_prompt)
        self.assertIn(command, summary_prompt)
        self.assertIn("Found 192.0.2.3:22", summary_prompt)
        self.assertNotIn("{instruction}", interpreter_prompt)
        self.assertNotIn("{new_context}", summary_prompt)

    def test_planner_rejects_nonboolean_or_unverified_goal_status(self):
        for response in (
            '{"steps": [], "goal_reached": "false", "goal_verification": "unknown"}',
            '{"steps": ["Run another check"], "goal_reached": true, "goal_verification": "already done"}',
            '{"steps": [], "goal_reached": true}',
            '{"steps": [], "goal_reached": false, "goal_verification": "needs a check"}',
        ):
            with self.assertRaises(ValueError):
                planner.parse_plan_response(response, "test")

    def test_memory_keeps_ports_without_raw_nmap_fingerprint(self):
        memory = InteractiveMemory()
        memory.record_action("find services", "nmap -sV host", {
            "output": "PORT STATE SERVICE\n22/tcp open ssh\nSF-Port2223-TCP:raw-data\nSF:more-raw-data\nNmap done",
            "exit_status": 0,
            "timed_out": False,
        })
        context = memory.context()
        self.assertIn("22/tcp open ssh", context)
        self.assertIn("Nmap done", context)
        self.assertNotIn("SF-Port2223", context)

    def test_saved_session_restores_goal_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = InteractiveMemory.new_session(directory)
            memory.start_goal("find nearby hosts")
            memory.record_action("find nearby hosts", "nmap -sn 172.20.0.1", {
                "output": "172.20.0.1 is up", "exit_status": 0, "timed_out": False,
            })
            restored = InteractiveMemory.load_session(memory.session_path, directory)
            self.assertEqual(restored.current_goal, "find nearby hosts")
            self.assertIn("172.20.0.1 is up", restored.context())
            restored.start_goal("inspect 172.20.0.1")
            self.assertEqual(InteractiveMemory.load_session("latest", directory).current_goal, "inspect 172.20.0.1")

    def test_legacy_experiment_imports_command_history(self):
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            experiment = Path(directory) / "experiment"
            experiment.mkdir()
            (experiment / "experiment.log").write_text("Goal: find hosts\n", encoding="utf-8")
            (experiment / "context.log").write_text(
                "Context Entry #1  2026-11-24 10:00:00\n"
                "    GOAL: find hosts\n"
                "    COMMAND EXECUTED: nmap -sn 172.20.0.1\n"
                "    COMMAND EXIT STATUS: 0\n"
                "    COMMAND TIMED OUT: no\n"
                "    COMMAND OUTPUT:\n"
                "    172.20.0.1 is up\n"
                "──────────────────────────────\n",
                encoding="utf-8",
            )
            restored = InteractiveMemory.load_session(experiment, Path(directory) / "sessions")
            self.assertEqual(restored.current_goal, "find hosts")
            self.assertIn("nmap -sn 172.20.0.1", restored.context())
            self.assertIn("172.20.0.1 is up", restored.context())
            self.assertTrue(restored.session_path.exists())

    def test_latest_imports_legacy_experiment_when_no_snapshot_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            experiment = Path(directory) / "experiments" / "old-run"
            experiment.mkdir(parents=True)
            (experiment / "experiment.log").write_text("Goal: list hosts\n", encoding="utf-8")
            (experiment / "context.log").write_text("Session Start\n", encoding="utf-8")
            memory = InteractiveMemory.load_session("latest", Path(directory) / "sessions")
            self.assertEqual(memory.current_goal, "list hosts")
            self.assertTrue(memory.session_path.exists())


if __name__ == "__main__":
    unittest.main()

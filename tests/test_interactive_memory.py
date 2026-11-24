import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from lib import planner
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
        response = '{"steps": ["Inspect 192.168.0.7"], "goal_reached": false}'
        with patch.object(planner, "call_model", return_value=response) as call:
            planner.generate_plan("Inspect the found host", prior_context=prior)
        self.assertIn(prior, call.call_args.args[0])
        self.assertIn("judge goal_reached only for the current goal", call.call_args.args[0])

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


if __name__ == "__main__":
    unittest.main()

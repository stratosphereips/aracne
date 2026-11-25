import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
if "paramiko" not in sys.modules and importlib.util.find_spec("paramiko") is None:
    sys.modules["paramiko"] = types.ModuleType("paramiko")

import aracne
from lib.command_approval import (
    confirmation_excerpt,
    request_human_confirmation,
    requests_human_confirmation,
)


class HumanConfirmationTests(unittest.TestCase):
    def test_detects_owner_approval_requests(self):
        for message in (
            "Run the scan and have the system owner confirm before continuing.",
            "Please get approval from the administrator.",
            "Awaiting human confirmation.",
            "Owner approval required before the next step.",
            "Ask the user to authorize this change.",
        ):
            self.assertTrue(requests_human_confirmation(message), message)
        for message in (
            "The system owner is alice.",
            "User confirmed the scan completed yesterday.",
            "The scan found an administrator login page.",
        ):
            self.assertFalse(requests_human_confirmation(message), message)

    def test_prompt_requires_explicit_yes_and_terminal(self):
        message = "Have the system owner confirm."
        self.assertFalse(request_human_confirmation(message, input_fn=lambda _: "yes", is_tty=False))
        self.assertFalse(request_human_confirmation(message, input_fn=lambda _: "y", is_tty=True))
        self.assertTrue(request_human_confirmation(message, input_fn=lambda _: "yes", is_tty=True))
        self.assertFalse(request_human_confirmation(message, input_fn=lambda _: (_ for _ in ()).throw(EOFError), is_tty=True))

    def test_excerpt_finds_request_after_large_output(self):
        output = ("unrelated output\n" * 100) + "Have the system owner confirm this scan before proceeding."
        self.assertIn("system owner confirm", confirmation_excerpt(output))
        self.assertLessEqual(len(confirmation_excerpt(output)), 500)

    def test_planner_request_asks_before_interpreter_and_continues_with_approval_context(self):
        client = types.SimpleNamespace(close=lambda: None)
        shell = types.SimpleNamespace(send=lambda value: None)
        plan_contexts = []

        def generate_plan(goal, context=None, fresh=False):
            plan_contexts.append(context)
            if fresh:
                return {"steps": ["Have the system owner confirm before checking the hostname."]}
            return {"steps": ["Check the hostname."]} if len(plan_contexts) == 2 else "goal reached"

        with patch.object(aracne, "connect_ssh", return_value=(client, shell)), patch.object(
            aracne, "generate_plan", side_effect=generate_plan
        ), patch.object(aracne.interpreter, "get_command_from_ollama", return_value="hostname") as interpret, patch.object(
            aracne, "request_human_confirmation", return_value=True
        ) as ask, patch.object(aracne, "approve_command", return_value=True), patch.object(
            aracne, "send_ssh_command", return_value={
                "output": "test-host\n", "exit_status": 0, "timed_out": False,
                "duration": 0.1, "error": None,
            }
        ) as send, patch.object(aracne, "conclude_session"), patch.object(
            aracne, "action_limit", 5
        ), patch.object(aracne, "context_history", []):
            aracne.execute_agent("Check hostname")

        ask.assert_called_once()
        interpret.assert_called_once_with("Check the hostname.")
        send.assert_called_once()
        self.assertEqual(send.call_args.args[1], "hostname")
        self.assertIn("User approved the requested action.", plan_contexts[1])

    def test_command_output_request_prompts_and_decline_stops_goal(self):
        client = types.SimpleNamespace(close=lambda: None)
        shell = types.SimpleNamespace(send=lambda value: None)
        output = "Result available. Have the system owner confirm before proceeding."
        with patch.object(aracne, "connect_ssh", return_value=(client, shell)), patch.object(
            aracne, "generate_plan", return_value={"steps": ["Check status"]}
        ) as generate, patch.object(
            aracne, "fetch_command_from_plan", return_value=("whoami", "Check status")
        ), patch.object(aracne, "request_human_confirmation", return_value=False) as ask, patch.object(
            aracne, "approve_command", return_value=True
        ), patch.object(aracne, "send_ssh_command", return_value={
            "output": output, "exit_status": 0, "timed_out": False,
            "duration": 0.1, "error": None,
        }) as send, patch.object(aracne, "conclude_session") as conclude, patch.object(
            aracne, "finalize_experiment"
        ) as finalize, patch.object(aracne, "action_limit", 5
        ), patch.object(aracne, "context_history", []):
            outcome = aracne.execute_agent("Check status")

        send.assert_called_once()
        ask.assert_called_once()
        generate.assert_called_once()
        conclude.assert_not_called()
        finalize.assert_called_once()
        self.assertEqual(outcome, "denied")

    def test_command_output_approval_is_passed_to_next_plan(self):
        client = types.SimpleNamespace(close=lambda: None)
        shell = types.SimpleNamespace(send=lambda value: None)
        contexts = []

        def generate_plan(goal, context=None, fresh=False):
            contexts.append(context)
            return {"steps": ["Check status"]} if fresh else "goal reached"

        with patch.object(aracne, "connect_ssh", return_value=(client, shell)), patch.object(
            aracne, "generate_plan", side_effect=generate_plan
        ), patch.object(
            aracne, "fetch_command_from_plan", return_value=("whoami", "Check status")
        ), patch.object(aracne, "request_human_confirmation", return_value=True) as ask, patch.object(
            aracne, "approve_command", return_value=True
        ), patch.object(aracne, "send_ssh_command", return_value={
            "output": "Please have the system owner confirm before proceeding.",
            "exit_status": 0, "timed_out": False, "duration": 0.1, "error": None,
        }), patch.object(aracne, "conclude_session") as conclude, patch.object(
            aracne, "action_limit", 5
        ), patch.object(aracne, "context_history", []):
            aracne.execute_agent("Check status")

        ask.assert_called_once()
        self.assertIn("User approved the requested action.", contexts[1])
        self.assertEqual(conclude.call_args.args[1], "success")


if __name__ == "__main__":
    unittest.main()

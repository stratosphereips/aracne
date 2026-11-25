import sys
import base64
import importlib.util
import subprocess
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
if "paramiko" not in sys.modules and importlib.util.find_spec("paramiko") is None:
    sys.modules["paramiko"] = types.ModuleType("paramiko")
import aracne
from lib import interpreter
from lib.memory import InteractiveMemory


class FakeShell:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.sent = []

    def send(self, text):
        self.sent.append(text)

    def recv_ready(self):
        return bool(self.chunks)

    def recv(self, _size):
        return self.chunks.pop(0)


class CommandExecutionTests(unittest.TestCase):
    def test_load_session_flag_defaults_to_latest(self):
        with patch.object(sys, "argv", ["aracne.py", "--load-session"]):
            self.assertEqual(aracne.parse_arguments().load_session, "latest")

    def test_final_summary_preserves_versions_and_all_facts(self):
        raw = (
            "- **22** — SSH (OpenSSH 10.2p1 Ubuntu 2ubuntu3.5)\n"
            "- **80** — HTTP (Python SimpleHTTPServer 0.6)\n"
            "- **2222** — SSH (OpenSSH 9.2p1 Debian)"
        )
        cleaned = aracne._clean_final_summary(raw)
        for fact in ("22", "10.2p1", "80", "0.6", "2222", "9.2p1"):
            self.assertIn(fact, cleaned)
        self.assertTrue(cleaned.startswith("22 — SSH"))

    def test_final_summary_formats_model_table_as_readable_rows(self):
        raw = (
            "The host exposes two services:\n"
            "| Port | Service | Detected version |\n"
            "|---|---|---|\n"
            "| 22/tcp | SSH | OpenSSH 10.2p1 Ubuntu |\n"
            "| 80/tcp | HTTP | SimpleHTTPServer 0.6 |"
        )
        cleaned = aracne._clean_final_summary(raw)
        self.assertIn("22/tcp: SSH (OpenSSH 10.2p1 Ubuntu)", cleaned)
        self.assertIn("80/tcp: HTTP (SimpleHTTPServer 0.6)", cleaned)
        self.assertNotIn("|---|", cleaned)

    def test_report_context_keeps_port_table_and_drops_fingerprint(self):
        previous = list(aracne.context_history)
        try:
            aracne.context_history[:] = [
                "GOAL: Find services\nCOMMAND: nmap -sV host\nCOMMAND OUTPUT:\n"
                "PORT STATE SERVICE VERSION\n22/tcp open ssh OpenSSH 10.2p1\n"
                + "SF:raw-fingerprint-data\n" * 700
                + "Nmap done: 1 IP address scanned\n"
            ]
            context = aracne._context_snapshot()
        finally:
            aracne.context_history[:] = previous
        self.assertLessEqual(len(context), 8000)
        self.assertIn("22/tcp open ssh OpenSSH 10.2p1", context)
        self.assertIn("Nmap done:", context)
        self.assertNotIn("SF:raw-fingerprint-data", context)

    def test_interactive_final_report_uses_prior_goal_memory(self):
        memory = InteractiveMemory()
        memory.record_action("find username", "whoami", {
            "output": "pedro", "exit_status": 0, "timed_out": False,
        })
        with patch.object(aracne, "call_model", return_value="The username is pedro.") as call:
            aracne.generate_final_report("What username was found?", "success", "Known from history", memory)
        self.assertIn("COMMAND: whoami", call.call_args.args[0])
        self.assertIn("pedro", call.call_args.args[0])

    def test_final_report_includes_its_instruction_and_preserves_braces(self):
        with patch.object(aracne, "context_history", ['GOAL: Inspect host\nCOMMAND OUTPUT: {"host":"192.0.2.3"}']), patch.object(
            aracne, "call_model", return_value="The host is 192.0.2.3."
        ) as call:
            aracne.generate_final_report("Inspect host", "success", "Planner marked goal reached")
        prompt = call.call_args.args[0]
        self.assertIn(aracne.FINAL_SUMMARY_CONFIG["instruction"], prompt)
        self.assertIn('{"host":"192.0.2.3"}', prompt)
        self.assertNotIn('{{"host"', prompt)
        self.assertEqual(call.call_args.kwargs["caller"], "interpreter.final_summary")

    def test_heredoc_is_sent_as_one_shell_line(self):
        command = "cat <<'EOF'\nScan documentation\nEOF"
        shell = FakeShell([b"__ARACNE_EXIT__2__0\r\n"])
        result = aracne.send_ssh_command(shell, command, 2, poll_interval=0)
        wire_command = shell.sent[0]
        self.assertEqual(len(wire_command.splitlines()), 1)
        self.assertNotIn("Scan documentation", wire_command)
        completed = subprocess.run(["sh", "-c", wire_command], capture_output=True, text=True, timeout=5)
        self.assertEqual(completed.returncode, 0)
        self.assertIn("Scan documentation", completed.stdout)
        self.assertIn("__ARACNE_EXIT__2__0", completed.stdout)
        self.assertEqual(result["exit_status"], 0)

    def test_interactive_interrupt_sends_remote_ctrl_c(self):
        class InterruptingShell(FakeShell):
            def __init__(self):
                super().__init__([])
                self.calls = 0

            def recv_ready(self):
                self.calls += 1
                if self.calls > 1:
                    raise KeyboardInterrupt
                return False

        shell = InterruptingShell()
        client = types.SimpleNamespace(close=lambda: None)
        with patch.object(aracne, "connect_ssh", return_value=(client, shell)), patch.object(
            aracne, "generate_plan", return_value={"steps": ["wait"]}
        ), patch.object(aracne, "fetch_command_from_plan", return_value=("sleep 30", "wait")), patch.object(
            aracne, "finalize_experiment"
        ) as finalize, patch.object(aracne, "conclude_session"), patch.object(aracne, "approve_command", return_value=True):
            aracne.action_limit = 2
            aracne.execute_agent("wait", interactive=True)
        self.assertIn("\x03", shell.sent)
        self.assertTrue(any(base64.b64encode(b"sleep 30").decode() in value for value in shell.sent))
        self.assertEqual(finalize.call_args.args[1], "interrupted")

    def test_interactive_mode_accepts_another_goal(self):
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal="first", ollama_ip=None,
                         summarize=False, goals_file="", interactive=True)
        with tempfile.TemporaryDirectory() as session_dir, patch.object(aracne, "SESSIONS_ROOT", Path(session_dir)), patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=(None, False, None, 2)
        ), patch.object(aracne, "verify_llm_connectivity"), patch.object(
            aracne, "print_llm_overview"
        ), patch.object(aracne, "update_logs"), patch.object(aracne, "execute_agent") as execute, patch(
            "builtins.input", side_effect=["second", "quit"]
        ):
            aracne.main()
        self.assertEqual([call.args[0] for call in execute.call_args_list], ["first", "second"])
        self.assertIs(execute.call_args_list[0].kwargs["memory"], execute.call_args_list[1].kwargs["memory"])

    def test_load_session_waits_for_new_goal_before_planning(self):
        memory = InteractiveMemory()
        memory.current_goal = "inspect a host"
        memory.recent = ["GOAL: find hosts\nOUTPUT:\n172.20.0.1 is up"]
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal=None, ollama_ip=None,
                         summarize=False, goals_file="", interactive=False, load_session="latest")
        with patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=(None, False, None, 2)
        ), patch.object(aracne.InteractiveMemory, "load_session", return_value=memory), patch.object(
            aracne, "verify_llm_connectivity"
        ), patch.object(aracne, "print_llm_overview"), patch.object(
            aracne, "update_logs"
        ), patch.object(aracne, "execute_agent") as execute, patch(
            "builtins.input", side_effect=["", "inspect 172.20.0.1", "quit"]
        ):
            aracne.main()
        self.assertEqual(execute.call_args.args[0], "inspect 172.20.0.1")
        self.assertIs(execute.call_args.kwargs["memory"], memory)
        self.assertEqual(memory.current_goal, "inspect 172.20.0.1")
        self.assertTrue(args.interactive)

    def test_load_session_quit_without_goal_does_not_call_model(self):
        memory = InteractiveMemory()
        memory.current_goal = "old goal"
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal=None, ollama_ip=None,
                         summarize=False, goals_file="", interactive=False, load_session="latest")
        with patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=("configured goal", False, None, 2)
        ), patch.object(aracne.InteractiveMemory, "load_session", return_value=memory), patch.object(
            aracne, "verify_llm_connectivity"
        ) as preflight, patch.object(aracne, "execute_agent") as execute, patch(
            "builtins.input", side_effect=["", "quit"]
        ):
            aracne.main()
        preflight.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(memory.current_goal, "old goal")

    def test_load_session_with_goal_starts_immediately(self):
        memory = InteractiveMemory()
        memory.current_goal = "old goal"
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal="new goal", ollama_ip=None,
                         summarize=False, goals_file="", interactive=False, load_session="latest")
        with patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=(None, False, None, 2)
        ), patch.object(aracne.InteractiveMemory, "load_session", return_value=memory), patch.object(
            aracne, "verify_llm_connectivity"
        ), patch.object(aracne, "print_llm_overview"), patch.object(
            aracne, "update_logs"
        ), patch.object(aracne, "execute_agent") as execute, patch(
            "builtins.input", return_value="quit"
        ) as user_input:
            aracne.main()
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0], "new goal")
        user_input.assert_called_once()

    def test_declined_command_returns_to_goal_prompt_without_interactive_flag(self):
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal="first", ollama_ip=None,
                         summarize=False, goals_file="", interactive=False, load_session=None)
        with tempfile.TemporaryDirectory() as session_dir, patch.object(
            aracne, "SESSIONS_ROOT", Path(session_dir)
        ), patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=(None, False, None, 2)
        ), patch.object(aracne, "verify_llm_connectivity"), patch.object(
            aracne, "print_llm_overview"
        ), patch.object(aracne, "update_logs"), patch.object(
            aracne, "execute_agent", side_effect=["denied", None]
        ) as execute, patch.object(
            aracne, "context_history", ["GOAL: first\nCOMMAND NOT EXECUTED: rm file"]
        ), patch("builtins.input", side_effect=["new instruction", "quit"]):
            aracne.main()
        self.assertEqual([call.args[0] for call in execute.call_args_list], ["first", "new instruction"])
        self.assertTrue(args.interactive)
        self.assertIn("COMMAND NOT EXECUTED", execute.call_args_list[1].kwargs["memory"].context())

    def test_interpreter_command_is_not_rewritten(self):
        generated = "nmap -sn -R 172.20.0.0/16"
        with patch.object(interpreter, "load_prompt", return_value="{instruction}"), patch.object(
            interpreter, "call_model", return_value=generated
        ):
            self.assertEqual(interpreter.get_command_from_ollama("find hosts"), generated)

    def test_nmap_gets_scan_timeout(self):
        self.assertEqual(aracne.command_timeout_seconds("nmap -sV 172.20.0.0/24"), 1800)
        self.assertEqual(aracne.command_timeout_seconds("echo ready"), 300)

    def test_scan_waits_for_exit_marker_and_preserves_output(self):
        shell = FakeShell([
            b"Starting Nmap\r\nNmap scan report for 172.20.0.1\r\n",
            b"__ARACNE_EXIT__2__0\r\n",
        ])
        result = aracne.send_ssh_command(shell, "nmap -sV 172.20.0.0/24", 2, poll_interval=0)
        self.assertEqual(result["exit_status"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["timeout_seconds"], 1800)
        self.assertIn("Nmap scan report", result["output"])

    def test_timeout_applies_even_when_output_keeps_arriving(self):
        class StreamingShell(FakeShell):
            def recv_ready(self):
                return True

            def recv(self, _size):
                return b"scan progress\r\n"

        shell = StreamingShell([])
        ticks = iter([0, 0, 0.5, 1.1, 1.1])
        with patch.object(aracne.time, "monotonic", side_effect=lambda: next(ticks)):
            result = aracne.send_ssh_command(shell, "nmap example.test", 2, timeout=1, poll_interval=0)
        self.assertTrue(result["timed_out"])
        self.assertIn("\x03", shell.sent)


if __name__ == "__main__":
    unittest.main()

import sys
import base64
import importlib.util
import subprocess
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
if importlib.util.find_spec("paramiko") is None:
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
        ) as finalize, patch.object(aracne, "conclude_session"):
            aracne.action_limit = 2
            aracne.execute_agent("wait", interactive=True)
        self.assertIn("\x03", shell.sent)
        self.assertTrue(any(base64.b64encode(b"sleep 30").decode() in value for value in shell.sent))
        self.assertEqual(finalize.call_args.args[1], "interrupted")

    def test_interactive_mode_accepts_another_goal(self):
        args = Namespace(env=".env", config="config/AttackConfig.yaml", goal="first", ollama_ip=None,
                         summarize=False, goals_file="", interactive=True)
        with patch.object(aracne, "parse_arguments", return_value=args), patch.object(
            aracne, "setup_env", return_value=(None, False, None, 2)
        ), patch.object(aracne, "verify_llm_connectivity"), patch.object(
            aracne, "print_llm_overview"
        ), patch.object(aracne, "update_logs"), patch.object(aracne, "execute_agent") as execute, patch(
            "builtins.input", side_effect=["second", "quit"]
        ):
            aracne.main()
        self.assertEqual([call.args[0] for call in execute.call_args_list], ["first", "second"])
        self.assertIs(execute.call_args_list[0].kwargs["memory"], execute.call_args_list[1].kwargs["memory"])

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

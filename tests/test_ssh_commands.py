import sys
import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
if importlib.util.find_spec("paramiko") is None:
    sys.modules["paramiko"] = types.ModuleType("paramiko")
import aracne
from lib.interpreter import normalize_command


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
    def test_nmap_python_formatter_is_replaced_by_direct_scan(self):
        generated = "nmap -n -sV 172.20.0.0/24 -oX - | python3 -c 'print(\\bad)'"
        self.assertEqual(normalize_command(generated), "nmap -n -sV 172.20.0.0/24")
        self.assertEqual(normalize_command("nmap -sV example.test"), "nmap -sV example.test")

    def test_host_discovery_disables_reverse_dns(self):
        generated = "nmap -sn -PR -R $(ip -o -4 route show scope link | awk '$1 ~ /^[0-9]/ {print $1}')"
        self.assertEqual(
            normalize_command(generated),
            "nmap -sn -PR -n $(ip -o -4 route show scope link | awk '$1 ~ /^[0-9]/ {print $1}')",
        )
        self.assertEqual(normalize_command("nmap -sn 192.168.0.0/24"), "nmap -n -sn 192.168.0.0/24")

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

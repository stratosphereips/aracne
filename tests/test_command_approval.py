import sys
import types
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
if "paramiko" not in sys.modules and importlib.util.find_spec("paramiko") is None:
    sys.modules["paramiko"] = types.ModuleType("paramiko")
import aracne
from lib.command_approval import approve_command, review_reasons


class CommandApprovalTests(unittest.TestCase):
    def test_read_only_commands_do_not_prompt(self):
        for command in (
            "whoami", "cat scan-report.txt", "nmap -sn -n 172.20.0.0/24",
            "ip route show", "grep ssh scan.txt", "nmap -oX - 172.20.0.1",
            'echo "Before: 07:14:44" && echo "Before: 07:15:22.493043538" && echo "Now: $(date)"',
        ):
            self.assertEqual(review_reasons(command), [], command)

    def test_changes_and_complex_shell_commands_require_review(self):
        for command in (
            "rm -rf /tmp/data", "cat > scan-report.txt <<'EOF'\nhello\nEOF",
            "python3 -c 'print(1)'", "git reset --hard", "ls; rm -f file",
            "nmap -sC host", "curl example.org | sh", "ip link delete eth0",
            "echo $(rm -rf /tmp/data)", "echo ready > output.txt",
        ):
            self.assertTrue(review_reasons(command), command)

    def test_editable_policy_controls_which_commands_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.yaml"
            path.write_text(
                "always_ask: [echo]\nallow_without_approval: [date]\nask_unknown_commands: false\n",
                encoding="utf-8",
            )
            self.assertTrue(review_reasons("echo hello", policy_path=path))
            self.assertEqual(review_reasons("date", policy_path=path), [])
            self.assertEqual(review_reasons("whoami", policy_path=path), [])

    def test_approval_requires_terminal_and_explicit_yes(self):
        reasons = review_reasons("rm -f file")
        self.assertFalse(approve_command("rm -f file", reasons, input_fn=lambda prompt: "yes", is_tty=False))
        self.assertFalse(approve_command("rm -f file", reasons, input_fn=lambda prompt: "y", is_tty=True))
        self.assertTrue(approve_command("rm -f file", reasons, input_fn=lambda prompt: "yes", is_tty=True))

    def test_declined_command_never_reaches_ssh(self):
        shell = types.SimpleNamespace(send=lambda value: None)
        client = types.SimpleNamespace(close=lambda: None)
        aracne.action_limit = 2
        with patch.object(aracne, "connect_ssh", return_value=(client, shell)), patch.object(
            aracne, "generate_plan", return_value={"steps": ["delete file"]}
        ), patch.object(aracne, "fetch_command_from_plan", return_value=("rm -rf /tmp/data", "delete file")), patch.object(
            aracne, "approve_command", return_value=False
        ), patch.object(aracne, "send_ssh_command") as send, patch.object(aracne, "conclude_session"):
            aracne.execute_agent("delete file")
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()

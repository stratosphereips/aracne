"""Bounded context carried between goals in an interactive session."""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from lib.utils import MODELS, call_model


def report_relevant_output(output):
    """Omit nmap's raw service fingerprint while retaining scan findings."""
    filtered = []
    omitted = False
    for line in output.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("SF:", "SF-")) or "service unrecognized despite returning data" in stripped:
            omitted = True
            continue
        filtered.append(line)
    if omitted:
        filtered.append("[Raw nmap service fingerprint omitted; full output is in logs.]")
    return "\n".join(filtered)


class InteractiveMemory:
    MAX_CONTEXT_CHARS = 9000
    MAX_OUTPUT_CHARS = 3000
    MAX_SUMMARY_CHARS = 3000

    def __init__(self, session_path=None):
        self.summary = ""
        self.recent = []
        self.current_goal = ""
        self.session_path = Path(session_path) if session_path else None

    @classmethod
    def new_session(cls, sessions_dir):
        sessions_dir = Path(sessions_dir)
        name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + ".json"
        return cls(sessions_dir / name)

    @classmethod
    def load_session(cls, path, sessions_dir):
        if str(path).lower() == "latest":
            candidates = list(Path(sessions_dir).glob("*.json"))
            if not candidates:
                candidates = list((Path(sessions_dir).parent / "experiments").glob("*/context.log"))
            if not candidates:
                raise FileNotFoundError(f"No saved sessions or experiments found near {sessions_dir}")
            path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
        path = Path(path).expanduser()
        if path.is_dir():
            if (path / "session.json").exists():
                path = path / "session.json"
            elif (path / "context.log").exists():
                return cls._load_experiment(path, sessions_dir)
            else:
                raise FileNotFoundError(f"No session.json or context.log found in {path}")
        elif path.name == "context.log":
            return cls._load_experiment(path.parent, sessions_dir)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("summary"), str)
            or not isinstance(data.get("recent"), list)
            or not all(isinstance(entry, str) for entry in data["recent"])
            or not isinstance(data.get("current_goal"), str)
        ):
            raise ValueError(f"Invalid session snapshot: {path}")
        memory = cls(path)
        memory.summary = data["summary"]
        memory.recent = data["recent"]
        memory.current_goal = data["current_goal"]
        return memory

    @classmethod
    def _load_experiment(cls, directory, sessions_dir):
        """Import a pre-snapshot experiment using its recorded command results."""
        context = (directory / "context.log").read_text(encoding="utf-8", errors="replace")
        context = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", context)
        header = re.compile(r"(?m)^Context Entry #\d+\s+\d{4}-\d{2}-\d{2}.*$")
        memory = cls()
        experiment_log = directory / "experiment.log"
        if experiment_log.exists():
            metadata = experiment_log.read_text(encoding="utf-8", errors="replace")
            goal_match = re.search(r"(?m)^Goal: (.+)$", metadata)
            if goal_match:
                memory.current_goal = goal_match.group(1)
        matches = list(header.finditer(context))
        for index, match in enumerate(matches):
            block = context[match.end():matches[index + 1].start() if index + 1 < len(matches) else None]
            block = re.split(r"(?m)^─{10,}\s*$", block, maxsplit=1)[0]
            goal = re.search(r"(?m)^\s{4}GOAL: (.*)$", block)
            command = re.search(r"(?m)^\s{4}COMMAND EXECUTED: (.*)$", block)
            status = re.search(r"(?m)^\s{4}COMMAND EXIT STATUS: (.*)$", block)
            timed_out = re.search(r"(?m)^\s{4}COMMAND TIMED OUT: (.*)$", block)
            output = re.search(r"(?ms)^\s{4}COMMAND OUTPUT:\s*\n(.*)", block)
            if not command:
                continue
            if goal:
                memory.current_goal = goal.group(1)
            status_text = status.group(1).strip() if status else ""
            memory.record_action(memory.current_goal, command.group(1), {
                "output": output.group(1) if output else "",
                "exit_status": int(status_text) if status_text.isdigit() else None,
                "timed_out": bool(timed_out and timed_out.group(1).strip() == "yes"),
            })
        if not memory.current_goal and not memory.recent:
            raise ValueError(f"No usable goal or command history in {directory}")
        memory.session_path = cls.new_session(sessions_dir).session_path
        memory.save()
        return memory

    def save(self):
        if self.session_path is None:
            return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "current_goal": self.current_goal,
            "summary": self.summary,
            "recent": self.recent,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.session_path.parent,
                prefix=self.session_path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.session_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _compress(self, text, purpose):
        prompt = (
            "Summarize verified facts for a continuing terminal session. "
            "Preserve exact IP addresses, hostnames, paths, commands, outcomes, and errors when useful. "
            "Do not invent facts or claim a new goal is already complete. "
            f"Keep the result under {self.MAX_SUMMARY_CHARS} characters.\n\n"
            f"Purpose: {purpose}\n\n{text}"
        )
        try:
            response = call_model(prompt, MODELS["planner"], caller="planner.memory_summary")
            if response and response.strip():
                return response.strip()[:self.MAX_SUMMARY_CHARS]
        except Exception:
            pass
        # A provider failure must not stop the interactive session.
        return text[-self.MAX_SUMMARY_CHARS:]

    def start_goal(self, goal):
        self.current_goal = goal
        if self.recent or self.summary:
            self.recent.append(f"NEW GOAL: {goal}")
            self._compact()
        self.save()

    def record_action(self, goal, command, result):
        output = report_relevant_output(result.get("output") or "<no output>")
        if len(output) > self.MAX_OUTPUT_CHARS:
            sample = output
            if len(output) > 12000:
                sample = output[:6000] + f"\n[... {len(output) - 12000} characters omitted ...]\n" + output[-6000:]
            output = self._compress(sample, "large command output")
        entry = (
            f"GOAL: {goal[:500]}\nCOMMAND: {command[:1000]}\n"
            f"EXIT STATUS: {result.get('exit_status')}\n"
            f"TIMED OUT: {result.get('timed_out', False)}\nOUTPUT:\n{output}"
        )
        self.recent.append(entry)
        self._compact()
        self.save()

    def _compact(self):
        while len(self.context()) > self.MAX_CONTEXT_CHARS and len(self.recent) > 1:
            oldest = self.recent.pop(0)
            self.summary = self._compress(
                f"Previous summary:\n{self.summary}\n\nOlder event:\n{oldest}",
                "cumulative memory across goals",
            )

    def context(self):
        pieces = []
        if self.summary:
            pieces.append("EARLIER SESSION SUMMARY:\n" + self.summary)
        if self.recent:
            pieces.append("RECENT SESSION ACTIONS:\n" + "\n\n".join(self.recent))
        return "\n\n".join(pieces)

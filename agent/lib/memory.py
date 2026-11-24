"""Bounded context carried between goals in an interactive session."""

from lib.utils import MODELS, call_model


class InteractiveMemory:
    MAX_CONTEXT_CHARS = 9000
    MAX_OUTPUT_CHARS = 3000
    MAX_SUMMARY_CHARS = 3000

    def __init__(self):
        self.summary = ""
        self.recent = []

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
        if self.recent or self.summary:
            self.recent.append(f"NEW GOAL: {goal}")
            self._compact()

    def record_action(self, goal, command, result):
        output = result.get("output") or "<no output>"
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

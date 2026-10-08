"""call_openai must send reproducible decoding parameters.

temperature defaults to 0 (override ARACNE_LLM_TEMPERATURE) and the seed
comes from ARACNE_LLM_SEED, so repeated calls issue identical requests
instead of sampling at the provider default.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.lib import utils  # noqa: E402


class _Msg:
    content = " out "


class _Resp:
    choices = [type("C", (), {"message": _Msg})]


class _Recorder:
    """Stand-in OpenAI client; records every create() kwargs."""

    def __init__(self):
        self.requests = []
        self.chat = type(
            "Chat", (), {"completions": type("Comp", (), {"create": self._create})()}
        )()

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return _Resp()


class LLMDeterminismTest(unittest.TestCase):
    def _call_twice(self, env):
        rec = _Recorder()
        with patch.dict(os.environ, env, clear=False), patch.object(
            utils, "_get_openai_client", return_value=rec
        ):
            utils.call_openai("cesnet", "prompt-a", "deepseek")
            utils.call_openai("cesnet", "prompt-b", "deepseek")
        return rec.requests

    def test_temperature_zero_by_default(self):
        reqs = self._call_twice({"ARACNE_LLM_SEED": ""})
        self.assertEqual(reqs[0]["temperature"], 0.0)

    def test_seed_from_env_and_identical_across_calls(self):
        reqs = self._call_twice({"ARACNE_LLM_SEED": "1337"})
        self.assertEqual(reqs[0]["seed"], 1337)
        # same decoding params on every call -> reproducible
        self.assertEqual(reqs[0]["seed"], reqs[1]["seed"])
        self.assertEqual(reqs[0]["temperature"], reqs[1]["temperature"])

    def test_no_seed_key_when_unset(self):
        reqs = self._call_twice({"ARACNE_LLM_SEED": ""})
        self.assertNotIn("seed", reqs[0])

    def test_temperature_override(self):
        reqs = self._call_twice({"ARACNE_LLM_TEMPERATURE": "0.7", "ARACNE_LLM_SEED": ""})
        self.assertEqual(reqs[0]["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()

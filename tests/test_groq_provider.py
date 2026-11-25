import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from lib import utils


class GroqProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "agent/config/AttackConfig.yaml").read_text()
        )
        self.config["default_provider"] = "groq"
        self.config["default_target"] = "ubuntu"

    def test_groq_uses_its_own_key_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(self.config))
            with patch.dict(os.environ, {"GROQ_API_KEY": "groq-test-key", "OPENAI_API_KEY": "openai-test-key"}):
                utils.load_environment(None, path)
                client = utils._get_openai_client("groq")
        self.assertEqual(utils.OPENAI_PROVIDER_KEYS["groq"], "groq-test-key")
        self.assertEqual(str(client.base_url), "https://api.groq.com/openai/v1/")
        self.assertEqual(utils.MODELS["planner"], "openai/gpt-oss-120b")

    def test_missing_groq_key_does_not_use_openai_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(self.config))
            with patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": "openai-test-key"}):
                with self.assertRaisesRegex(ValueError, "Missing GROQ_API_KEY"):
                    utils.load_environment(None, path)


if __name__ == "__main__":
    unittest.main()

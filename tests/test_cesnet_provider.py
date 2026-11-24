import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from lib import utils


class CesnetProviderTests(unittest.TestCase):
    def _config_path(self, directory, base_url):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "agent/config/AttackConfig.yaml").read_text())
        config["planner_source"] = "cesnet"
        config["interpreter_source"] = "cesnet"
        config["planner_model"] = "glm-5.3"
        config["interpreter_model"] = "glm-5.3"
        config["llm_providers"]["cesnet"]["base_url"] = base_url
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_legacy_endpoint_uses_documented_api_url(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CESNET_API_KEY": "test-key"}):
            path = self._config_path(directory, utils.CESNET_LEGACY_BASE_URL)
            utils.load_environment(None, path)
        self.assertEqual(utils.LLM_PROVIDERS["cesnet"]["base_url"], utils.CESNET_BASE_URL)
        self.assertEqual(utils.OPENAI_PROVIDER_KEYS["cesnet"], "test-key")

    def test_cesnet_key_does_not_fall_back_to_openai_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CESNET_API_KEY": "", "OPENAI_API_KEY": "openai-only"}
        ):
            path = self._config_path(directory, utils.CESNET_BASE_URL)
            with self.assertRaisesRegex(ValueError, "Missing CESNET_API_KEY"):
                utils.load_environment(None, path)

    def test_preflight_checks_actual_completion(self):
        client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="glm-5.3")])))
        with patch.object(utils, "_get_openai_client", return_value=client), patch.object(
            utils, "call_openai", return_value="OK"
        ) as completion:
            utils._ensure_openai_models_available("cesnet", {"glm-5.3"})
        completion.assert_called_once_with("cesnet", "Reply OK.", "glm-5.3", temperature=1.0)

    def test_preflight_reports_completion_denial(self):
        client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="glm-5.3")])))
        with patch.object(utils, "_get_openai_client", return_value=client), patch.object(
            utils, "call_openai", side_effect=utils.LLMProviderError("cesnet", "HTTP 403")
        ):
            with self.assertRaisesRegex(RuntimeError, "chat completion failed: HTTP 403"):
                utils._ensure_openai_models_available("cesnet", {"glm-5.3"})

    def test_preflight_rejects_unlisted_model_without_completion(self):
        client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[])))
        with patch.object(utils, "_get_openai_client", return_value=client), patch.object(utils, "call_openai") as completion:
            with self.assertRaisesRegex(RuntimeError, "not listed for this CESNET API key"):
                utils._ensure_openai_models_available("cesnet", {"glm-5.3"})
        completion.assert_not_called()

    def test_403_error_identifies_cesnet_access_problem(self):
        def deny(**kwargs):
            raise RuntimeError("Error code: 403 - permission denied")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=deny)))
        with patch.object(utils, "_get_openai_client", return_value=client):
            with self.assertRaisesRegex(utils.LLMProviderError, "CESNET denied chat access to model 'glm-5.3'"):
                utils.call_openai("cesnet", "hello", "glm-5.3", temperature=1.0)


if __name__ == "__main__":
    unittest.main()

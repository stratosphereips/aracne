import json
import logging
import os
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional
import re

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI

OPENAI_API_KEY = ""
SSH_CONFIG: Dict[str, Optional[str]] = {}
MODELS = {
    "interpreter": "llama3.2",
    "planner": "o3-mini-2025-01-31",
    "summarizer": "gpt-4o-2024-08-06",
}
PLANNER_SOURCE = "openai"
SUMMARIZER_SOURCE = "openai"
INTERPRETER_SOURCE = "local"
OLLAMA_URL = ""
OPENAI_CLIENT: OpenAI | None = None
OPENAI_BASE_URL: str | None = None
OLLAMA_SESSION: requests.Session | None = None
PROMPT_CACHE: Dict[Path, str] = {}
PROMPT_DICT_CACHE: Dict[Path, dict] = {}
LLM_PROVIDERS: Dict[str, Dict[str, object]] = {}
OPENAI_CLIENTS: Dict[str, OpenAI] = {}
OPENAI_PROVIDER_KEYS: Dict[str, str] = {}
OLLAMA_BASE_URL: str | None = None

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SHELL_LOG_PATH = LOGS_DIR / "agent.log"
LOG_FILE = SHELL_LOG_PATH  # Backwards compatibility for existing imports
CONTEXT_LOG_PATH = LOGS_DIR / "context.log"

COMMAND_COLOR = "\033[1;32m"
OUTPUT_COLOR = "\033[0;36m"
HEADER_COLOR = "\033[1;35m"
TITLE_COLOR = "\033[1;33m"
LABEL_COLOR = "\033[1;36m"
SECTION_COLOR = "\033[1;34m"
TEXT_COLOR = "\033[0;37m"
SUCCESS_COLOR = "\033[1;32m"
ERROR_COLOR = "\033[1;31m"
TIMESTAMP_COLOR = "\033[90m"
RESET_COLOR = "\033[0m"

_CALL_SEQUENCE = count(1)


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider encounters an unrecoverable problem."""

    def __init__(self, provider: str, message: str):
        super().__init__(message)
        self.provider = provider

def _configure_logging():
    """Configure dedicated loggers for shell simulation and context tracing."""
    global logger, shell_logger, context_logger

    if getattr(_configure_logging, "configured", False):
        return

    shell_handler = logging.FileHandler(SHELL_LOG_PATH, mode="w", encoding="utf-8")
    shell_handler.setFormatter(logging.Formatter("%(message)s"))

    context_handler = logging.FileHandler(CONTEXT_LOG_PATH, mode="w", encoding="utf-8")
    context_handler.setFormatter(logging.Formatter("%(message)s"))

    shell_logger = logging.getLogger("aracne.shell")
    shell_logger.setLevel(logging.INFO)
    shell_logger.handlers.clear()
    shell_logger.addHandler(shell_handler)
    shell_logger.propagate = False

    context_logger = logging.getLogger("aracne.context")
    context_logger.setLevel(logging.INFO)
    context_logger.handlers.clear()
    context_logger.addHandler(context_handler)
    context_logger.propagate = False

    # Maintain backwards compatibility with modules importing `logger`
    logger = context_logger

    _configure_logging.configured = True


_configure_logging()


def reset_call_sequence():
    """Reset the LLM call counter (used when starting a new run)."""
    global _CALL_SEQUENCE
    _CALL_SEQUENCE = count(1)


def _colorize(payload: str, color: str) -> str:
    return f"{color}{payload}{RESET_COLOR}"


def log_shell_command(command: str):
    """Record a command as shown in the simulated remote shell."""
    if command is None:
        return
    shell_logger.info(_colorize(f"$ {command}", COMMAND_COLOR))


def log_shell_output(output: str):
    """Record command output in the simulated remote shell log."""
    text = (output or "").rstrip("\n")
    if not text.strip():
        text = "<no output>"
    shell_logger.info(_colorize(text, OUTPUT_COLOR))


def _divider(char: str = "═", width: int = 78) -> str:
    return char * width


def _indent_block(text: str, indent: str = "    ") -> str:
    lines = text.splitlines() or [""]
    return "\n".join(f"{indent}{line}" if line else indent for line in lines)


def log_context_event(title: str, body: str, color: str = SECTION_COLOR):
    """Log a formatted context event with separators and timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "\n".join(
        [
            _colorize(_divider("═"), HEADER_COLOR),
            f"{_colorize(title, color)}  {_colorize(timestamp, TIMESTAMP_COLOR)}",
            _colorize(_indent_block(body), TEXT_COLOR),
            _colorize(_divider("─"), HEADER_COLOR),
            "",
        ]
    )
    context_logger.info(message)
    return message


def _extract_json_block(text: str) -> str | None:
    start = None
    stack: list[str] = []
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if start is None:
            if ch in '{[':
                start = idx
                stack.append('}' if ch == '{' else ']')
        else:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in '{[':
                    stack.append('}' if ch == '{' else ']')
                elif stack and ch == stack[-1]:
                    stack.pop()
                    if not stack:
                        return text[start:idx + 1]
        if ch != '\\' and not in_string:
            escape = False
    return None


def normalize_llm_json(raw: str) -> str:
    """Best-effort cleanup to coerce LLM output into valid JSON."""
    if raw is None:
        raise ValueError("No response to normalise")

    work = raw.strip()
    if not work:
        raise ValueError("Empty response from LLM")

    work = re.sub(r'^```(?:json)?', '', work, flags=re.IGNORECASE).strip()
    work = re.sub(r'```$', '', work).strip()

    block = _extract_json_block(work)
    if block:
        work = block.strip()

    attempts = [work]

    trailing_clean = re.sub(r',\s*(?=[}\]])', '', work)
    if trailing_clean != work:
        attempts.append(trailing_clean)

    single_quote_clean = work.replace("'", '"')
    if single_quote_clean != work:
        attempts.append(single_quote_clean)

    for candidate in attempts:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    # As a last resort return the original block; caller will raise with context.
    return work


def _log_llm_start(caller: str, source: str, model: str, prompt: str, instruction: Optional[str]):
    call_id = next(_CALL_SEQUENCE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = _colorize(_divider("═"), HEADER_COLOR)
    title_line = (
        f"{_colorize(f'LLM Call {call_id}', TITLE_COLOR)}  "
        f"{_colorize(timestamp, TIMESTAMP_COLOR)}"
    )
    meta = "\n".join(
        [
            f"  {_colorize('Trigger:', LABEL_COLOR)} {_colorize(caller, TEXT_COLOR)}",
            f"  {_colorize('Source :', LABEL_COLOR)} {_colorize(source, TEXT_COLOR)}",
            f"  {_colorize('Model  :', LABEL_COLOR)} {_colorize(model, TEXT_COLOR)}",
        ]
    )
    prompt_block = "\n".join(
        [
            f"{_colorize('Prompt:', SECTION_COLOR)}",
            _colorize(_indent_block(prompt), TEXT_COLOR),
        ]
    )

    blocks = [header, title_line, meta, prompt_block]

    if instruction is not None:
        blocks.append(
            "\n".join(
                [
                    f"{_colorize('Instruction:', SECTION_COLOR)}",
                    _colorize(_indent_block(instruction), TEXT_COLOR),
                ]
            )
        )

    context_logger.info("\n".join(blocks))
    return call_id


def _log_llm_success(call_id: int, response: str, duration: float | None = None):
    pretty_response = response
    try:
        parsed = json.loads(response)
        pretty_response = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pass

    lines = [
        f"{_colorize('Response:', SECTION_COLOR)}",
        _colorize(_indent_block(pretty_response), SUCCESS_COLOR),
    ]
    if duration is not None:
        lines.insert(
            1,
            f"{_colorize('Duration:', LABEL_COLOR)} {_colorize(f'{duration:.2f}s', TEXT_COLOR)}",
        )
    lines.extend(
        [
            _colorize(_divider("─"), HEADER_COLOR),
            "",
        ]
    )
    context_logger.info("\n".join(lines))


def _log_llm_failure(call_id: int, error: Exception, duration: float | None = None):
    lines = [
        f"{_colorize('Error:', SECTION_COLOR)}",
        _colorize(_indent_block(str(error)), ERROR_COLOR),
    ]
    if duration is not None:
        lines.insert(
            1,
            f"{_colorize('Duration:', LABEL_COLOR)} {_colorize(f'{duration:.2f}s', TEXT_COLOR)}",
        )
    lines.extend(
        [
            _colorize(_divider("─"), HEADER_COLOR),
            "",
        ]
    )
    context_logger.info("\n".join(lines))


def load_prompt(path):
    """Loads the summarization prompt from .yml."""
    prompt_path = Path(path)

    if not prompt_path.is_absolute():
        for base in (AGENT_ROOT, REPO_ROOT, Path.cwd()):
            candidate = base / prompt_path
            if candidate.exists():
                prompt_path = candidate
                break
        else:
            raise FileNotFoundError(f"Prompt file not found at {prompt_path} from agent or repository roots.")

    resolved_path = prompt_path.resolve()
    cached_prompt = PROMPT_CACHE.get(resolved_path)
    if cached_prompt is not None:
        return cached_prompt

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    prompt = config["prompt"]
    PROMPT_CACHE[resolved_path] = prompt
    return prompt


def load_prompt_config(path):
    """Load a YAML prompt configuration file and cache the resulting dictionary."""
    prompt_path = Path(path)

    if not prompt_path.is_absolute():
        for base in (AGENT_ROOT, REPO_ROOT, Path.cwd()):
            candidate = base / prompt_path
            if candidate.exists():
                prompt_path = candidate
                break
        else:
            raise FileNotFoundError(f"Prompt config not found at {prompt_path} from agent or repository roots.")

    resolved_path = prompt_path.resolve()
    cached_config = PROMPT_DICT_CACHE.get(resolved_path)
    if cached_config is not None:
        return cached_config

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Prompt config at {resolved_path} must be a mapping.")

    PROMPT_DICT_CACHE[resolved_path] = config
    return config


def _resolve_provider_kind(provider_name: str) -> str:
    name = (provider_name or "").strip().lower()
    provider = LLM_PROVIDERS.get(name, {}) if name else {}
    kind = (provider.get("kind") if isinstance(provider, dict) else None) or name
    mapping = {
        "openai": "openai",
        "cesnet": "openai",
        "local": "ollama",
        "ollama": "ollama",
    }
    resolved = mapping.get(kind, kind)
    if resolved not in {"openai", "ollama"}:
        raise ValueError(f"Error: unsupported provider kind '{kind}' for '{provider_name}'.")
    return resolved


def _resolve_config_path(config_path: str | Path | None) -> Path:
    config_path_obj = Path(config_path) if config_path else AGENT_ROOT / "config" / "AttackConfig.yaml"

    if not config_path_obj.is_absolute():
        for base in (REPO_ROOT, AGENT_ROOT, Path.cwd()):
            candidate = base / config_path_obj
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Error: The specified config file does not exist: {config_path}")
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Error: The specified config file does not exist: {config_path_obj}")
    return config_path_obj


def _select_target(config: dict) -> dict:
    targets = config.get("targets")
    if not targets:
        raise ValueError("Error: No SSH targets defined under 'targets' in the configuration.")

    default_key = config.get("default_target")
    if not default_key:
        raise ValueError("Error: 'default_target' is not specified in the configuration.")

    target = targets.get(default_key)
    if not target:
        available = ", ".join(sorted(targets))
        raise ValueError(f"Error: default_target '{default_key}' not found. Available targets: {available}")

    required_fields = ("host", "port", "user", "password")
    missing = [field for field in required_fields if field not in target or target[field] is None]
    if missing:
        raise ValueError(f"Error: Target '{default_key}' is missing required fields: {', '.join(missing)}")

    return target


def _get_provider(name: str) -> Dict[str, object]:
    key = (name or "").strip().lower()
    provider = LLM_PROVIDERS.get(key)
    if provider is None:
        provider = {"kind": key or "openai"}
        LLM_PROVIDERS[key] = provider
    return provider


def load_environment(env_path, config_path=None):
    """Loads environment variables and attack configuration."""
    global OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_CLIENT, OPENAI_CLIENTS
    global SSH_CONFIG, MODELS, PLANNER_SOURCE, SUMMARIZER_SOURCE, INTERPRETER_SOURCE
    global OLLAMA_URL, OLLAMA_BASE_URL, LLM_PROVIDERS, OPENAI_PROVIDER_KEYS

    OPENAI_CLIENT = None
    OPENAI_CLIENTS.clear()
    OPENAI_PROVIDER_KEYS.clear()
    OLLAMA_BASE_URL = None
    OLLAMA_URL = ""

    if env_path:
        env_path_obj = Path(env_path)
        if not env_path_obj.is_absolute():
            for base in (REPO_ROOT, AGENT_ROOT, Path.cwd()):
                candidate = base / env_path_obj
                if candidate.exists():
                    env_path_obj = candidate
                    break
            else:
                raise FileNotFoundError(f"Error: The specified .env file does not exist: {env_path}")
        elif not env_path_obj.exists():
            raise FileNotFoundError(f"Error: The specified .env file does not exist: {env_path_obj}")
        load_dotenv(env_path_obj)

    config_path_obj = _resolve_config_path(config_path)

    with open(config_path_obj, "r", encoding="utf-8") as f:
        attack_config = yaml.safe_load(f)

    LLM_PROVIDERS = {
        (name or "").strip().lower(): dict(config or {})
        for name, config in (attack_config.get("llm_providers") or {}).items()
    }

    goal = attack_config["goal"]
    target_profile = _select_target(attack_config)
    SSH_CONFIG = {
        "host": target_profile["host"],
        "port": int(target_profile.get("port", 22)),
        "user": target_profile["user"],
        "password": target_profile["password"],
    }
    interpreter_model = attack_config["interpreter_model"]
    planner_model = attack_config["planner_model"]
    summarizer_model = attack_config["summarizer_model"]
    summarize = attack_config["summarizing"]
    action_limit = attack_config["action_limit"]

    planner_source_raw = attack_config["planner_source"].strip().lower()
    summarizer_source_raw = attack_config["summarizer_source"].strip().lower()
    interpreter_source_raw = attack_config["interpreter_source"].strip().lower()

    PLANNER_SOURCE = planner_source_raw
    SUMMARIZER_SOURCE = summarizer_source_raw
    INTERPRETER_SOURCE = interpreter_source_raw

    used_providers = {
        planner_source_raw,
        summarizer_source_raw,
        interpreter_source_raw,
    }

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

    for provider_name in used_providers:
        provider = _get_provider(provider_name)
        kind = _resolve_provider_kind(provider_name)
        if kind == "openai":
            env_var = provider.get("api_key_env")
            key = provider.get("api_key")
            if env_var:
                key = key or os.getenv(str(env_var))
            if not key and provider_name == "cesnet":
                key = os.getenv("CESNET_API_KEY")
            if not key and not OPENAI_API_KEY:
                raise ValueError(f"Error: Missing API key for provider '{provider_name}'.")
            provider_key = key or OPENAI_API_KEY
            if not provider_key:
                raise ValueError(f"Error: Missing API key for provider '{provider_name}'.")
            provider["api_key"] = provider_key
            OPENAI_PROVIDER_KEYS[provider_name] = provider_key
            base_url = provider.get("base_url") or provider.get("url")
            if not base_url and provider_name == "cesnet":
                base_url = "https://chat.ai.e-infra.cz/api"
            if not base_url and provider_name == "openai":
                base_url = OPENAI_BASE_URL
            if base_url:
                provider["base_url"] = str(base_url).rstrip("/")
                if provider_name == "openai" and not OPENAI_BASE_URL:
                    OPENAI_BASE_URL = provider["base_url"]
            OPENAI_API_KEY = provider_key if provider_name == "openai" else OPENAI_API_KEY or provider_key
        elif kind == "ollama":
            base_url = provider.get("base_url") or provider.get("url")
            host = provider.get("host") or attack_config.get("ollama_url") or OLLAMA_URL
            port = provider.get("port") or provider.get("http_port") or 11434
            if not base_url:
                if not host:
                    raise ValueError(f"Error: Host not specified for provider '{provider_name}'.")
                base_url = f"http://{host}:{port}"
            provider["base_url"] = str(base_url).rstrip("/")
            if provider_name == "ollama":
                OLLAMA_BASE_URL = provider["base_url"]
                OLLAMA_URL = host or OLLAMA_URL
    requires_openai = any(
        _resolve_provider_kind(name) == "openai"
        for name in used_providers
    )
    if requires_openai and not (OPENAI_API_KEY or OPENAI_PROVIDER_KEYS):
        raise ValueError("Error: Missing OpenAI-compatible API key for selected providers.")

    if any(_resolve_provider_kind(name) == "ollama" for name in used_providers) and not (
        OLLAMA_BASE_URL or OLLAMA_URL
    ):
        raise ValueError("Error: Ollama provider selected but no base URL/host configured.")

    # Update the global MODELS dictionary
    MODELS["interpreter"] = interpreter_model
    MODELS["planner"] = planner_model
    MODELS["summarizer"] = summarizer_model

    return goal, summarize, OLLAMA_BASE_URL or OLLAMA_URL, action_limit


def _get_openai_client(provider_name: str) -> OpenAI:
    provider = _get_provider(provider_name)
    api_key = provider.get("api_key") or OPENAI_PROVIDER_KEYS.get(provider_name) or OPENAI_API_KEY
    if not api_key:
        raise RuntimeError(f"OpenAI-compatible provider '{provider_name}' is missing an API key.")
    client = OPENAI_CLIENTS.get(provider_name)
    if client is None:
        kwargs = {"api_key": api_key}
        base_url = provider.get("base_url") or OPENAI_BASE_URL
        if base_url:
            kwargs["base_url"] = str(base_url).rstrip("/")
        client = OpenAI(**kwargs)
        OPENAI_CLIENTS[provider_name] = client
    return client


def call_openai(provider_name: str, prompt: str, model: str):
    """Calls an OpenAI-compatible API provider with the given prompt."""
    client = _get_openai_client(provider_name)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as exc:
        message = str(exc).lower()
        if provider_name == "cesnet" and ("not found" in message or "sleep" in message):
            raise LLMProviderError(
                provider_name,
                "Requested model is unavailable or the CESNET service is sleeping. "
                "Please verify the model name (e.g. 'qwen3-coder') or wake the service.",
            ) from exc
        raise LLMProviderError(provider_name, f"OpenAI-compatible request failed: {exc}") from exc


def _build_chat_messages(prompt: str, instruction: str | None) -> List[Dict[str, str]]:
    if instruction:
        return [
            {"role": "system", "content": str(prompt)},
            {"role": "user", "content": str(instruction)},
        ]
    return [{"role": "user", "content": str(prompt)}]


def call_local(prompt, model, instruction=None, base_url: str | None = None, provider_name: str | None = None):
    provider_label = provider_name or "ollama"
    base = base_url or OLLAMA_BASE_URL
    if not base:
        host = OLLAMA_URL
        if not host:
            raise LLMProviderError(provider_label, "Ollama base URL is not configured.")
        base = f"http://{host}:11434"
    base = base.rstrip("/")
    payload = {
        "model": model,
        "messages": _build_chat_messages(prompt, instruction),
        "stream": False,
    }

    attempt_errors: list[str] = []

    for attempt in range(1, 11):
        try:
            response = requests.post(f"{base}/api/chat", json=payload, timeout=480)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.HTTPError as http_exc:
            detail = ""
            try:
                detail = http_exc.response.json()
            except Exception:
                detail = http_exc.response.text if http_exc.response is not None else ""

            if http_exc.response is not None and http_exc.response.status_code == 404:
                raise LLMProviderError(
                    provider_label,
                    f"Ollama model '{model}' not available at {base}. Detail: {detail}",
                ) from http_exc
            raise LLMProviderError(provider_label, f"Ollama HTTP error. Detail: {detail}") from http_exc
        except requests.exceptions.RequestException as req_exc:
            attempt_errors.append(str(req_exc))
            if attempt < 10:
                time.sleep(2)
                continue
            raise LLMProviderError(
                provider_label,
                f"Failed to reach Ollama at {base} after {attempt} attempts. Errors: {attempt_errors}",
            ) from req_exc
    else:
        raise LLMProviderError(
            provider_label,
            f"Exhausted retries connecting to Ollama at {base}. Errors: {attempt_errors}",
        )

    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise LLMProviderError(provider_label, "Ollama returned an empty response.")
    return content


def call_model(prompt, model, instruction=None, caller: str = "unspecified"):
    """
    Calls the specified model with the given prompt and returns the response. Any model and any source is possible.
    """
    provider_name = None
    formatted_prompt = prompt
    display_instruction = None

    if model == MODELS["planner"]:
        provider_name = PLANNER_SOURCE
    elif model == MODELS["summarizer"]:
        provider_name = SUMMARIZER_SOURCE
    elif model == MODELS["interpreter"]:
        provider_name = INTERPRETER_SOURCE
        if _resolve_provider_kind(provider_name) == "openai" and instruction is not None:
            formatted_prompt = prompt.format(instruction=instruction)
        else:
            display_instruction = instruction
    else:
        raise ValueError(f"Error: Model '{model}' is not recognised in MODELS.")

    provider_kind = _resolve_provider_kind(provider_name)

    if provider_kind == "ollama" and display_instruction is None:
        display_instruction = instruction

    call_id = _log_llm_start(
        caller=caller,
        source=provider_name,
        model=model,
        prompt=formatted_prompt if provider_kind == "openai" else prompt,
        instruction=display_instruction,
    )

    try:
        start_ts = time.perf_counter()
        if provider_kind == "openai":
            response = call_openai(provider_name, formatted_prompt, model)
        elif provider_kind == "ollama":
            provider = _get_provider(provider_name)
            base_url = provider.get("base_url") or OLLAMA_BASE_URL
            response = call_local(prompt, model, instruction, base_url=base_url, provider_name=provider_name)
        else:
            raise ValueError(f"Error: Unsupported provider '{provider_name}' for model '{model}'.")
        duration = time.perf_counter() - start_ts
        _log_llm_success(call_id, response, duration)
        return response
    except Exception as exc:
        duration = time.perf_counter() - start_ts
        _log_llm_failure(call_id, exc, duration)
        if isinstance(exc, LLMProviderError):
            raise
        raise LLMProviderError(
            provider_name,
            f"LLM provider '{provider_name}' failed while handling '{caller}': {exc}",
        ) from exc


def _ensure_openai_models_available(provider_name: str, models: set[str]):
    client = _get_openai_client(provider_name)
    failures: list[str] = []

    for model in sorted(models):
        try:
            client.models.retrieve(model=model)
        except Exception as exc:  # noqa: BLE001 - we want full context to bubble up
            failures.append(f"{model}: {exc}")

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"OpenAI connectivity check failed for: {joined}")


def _ensure_local_models_available(base_url: str, models: set[str]):
    base = base_url.rstrip("/") if base_url else None
    if not base:
        raise RuntimeError("Ollama base URL is not configured.")
    try:
        response = requests.get(f"{base}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama connectivity check failed: {exc}") from exc

    data = response.json()
    available = {
        item.get("name")
        for item in data.get("models", [])
        if item.get("name")
    }
    missing = sorted(model for model in models if model not in available)
    if missing:
        raise RuntimeError(
            f"Ollama models not available on {base}: {', '.join(missing)}"
        )


def verify_llm_connectivity(require_summarizer: bool = False):
    """Ensure configured LLM backends are reachable before shell actions."""
    roles = {
        "planner": (PLANNER_SOURCE, MODELS["planner"]),
        "interpreter": (INTERPRETER_SOURCE, MODELS["interpreter"]),
    }
    if require_summarizer:
        roles["summarizer"] = (SUMMARIZER_SOURCE, MODELS["summarizer"])

    openai_models: Dict[str, set[str]] = {}
    local_models: Dict[str, set[str]] = {}

    for source_name, model in roles.values():
        kind = _resolve_provider_kind(source_name)
        if kind == "openai":
            openai_models.setdefault(source_name, set()).add(model)
        elif kind == "ollama":
            local_models.setdefault(source_name, set()).add(model)

    failures: list[str] = []

    for provider_name, models in openai_models.items():
        try:
            _ensure_openai_models_available(provider_name, models)
        except RuntimeError as exc:
            failures.append(f"{provider_name}: {exc}")

    for provider_name, models in local_models.items():
        provider = _get_provider(provider_name)
        base_url = provider.get("base_url") or OLLAMA_BASE_URL
        if not base_url:
            host = provider.get("host") or OLLAMA_URL
            port = provider.get("port", 11434)
            if host:
                base_url = f"http://{host}:{port}"
        if not base_url:
            failures.append(f"{provider_name}: missing Ollama base URL configuration")
            continue
        try:
            _ensure_local_models_available(base_url, models)
        except RuntimeError as exc:
            failures.append(f"{provider_name}: {exc}")

    if failures:
        message = "LLM connectivity preflight failed:\n" + "\n".join(
            f"- {failure}" for failure in failures
        )
        logger.error(message)
        raise RuntimeError(message)

    logger.info("LLM connectivity preflight checks passed.")


def get_llm_configuration() -> Dict[str, Dict[str, str]]:
    """Return a snapshot of current LLM routing configuration."""
    return {
        "planner": {
            "provider": PLANNER_SOURCE,
            "kind": _resolve_provider_kind(PLANNER_SOURCE),
            "model": MODELS["planner"],
        },
        "interpreter": {
            "provider": INTERPRETER_SOURCE,
            "kind": _resolve_provider_kind(INTERPRETER_SOURCE),
            "model": MODELS["interpreter"],
        },
        "summarizer": {
            "provider": SUMMARIZER_SOURCE,
            "kind": _resolve_provider_kind(SUMMARIZER_SOURCE),
            "model": MODELS["summarizer"],
        },
    }

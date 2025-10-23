# ARACNE: Autonomous LLM-based Pentesting Agent

ARACNE is an SSH-driven pentesting agent orchestrated by LLMs. It can plan, execute, and adapt attack chains while recording detailed telemetry across multiple logs. It supports both cloud-hosted and local (Ollama) models with per-role configuration.

## Table of Contents

- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Goals](#goals)
- [Logs](#logs)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Setup

### Requirements

- Python 3.10+
- `pip`
- Optional: Docker, Ollama daemon

### Installation

```bash
git clone https://github.com/stratosphereips/aracne.git
cd aracne

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp env_EXAMPLE agent/.env
```

Populate `agent/.env` with SSH target credentials and (optional) OpenAI API key.

## Configuration

Runtime behavior is driven by YAML files (default: `agent/config/AttackConfig.yaml`). Use `--config` to point to alternatives (e.g., `AttackConfig_local.yaml`).

```yaml
goal: ""                     # Optional default goal
llm_providers:
  openai:
    kind: openai            # Uses OPENAI_API_KEY / OPENAI_BASE_URL if set
  ollama:
    kind: ollama
    host: localhost
    port: 11434
  cesnet:
    kind: openai
    base_url: https://chat.ai.e-infra.cz/api
    api_key_env: CESNET_API_KEY
targets:                     # Named SSH target profiles
  ubuntu:
    host: localhost
    port: 23
    user: root
    password: 123456
  shellm:
    host: localhost
    port: 22
    user: admin
    password: admin
default_target: ubuntu        # Which profile to load by default
planner_source: "openai"     # openai | ollama | cesnet
planner_model: "o3-mini-2025-01-31"
summarizing: false
summarizer_source: "openai"  # openai | ollama | cesnet
summarizer_model: "gpt-4o-2024-08-06"
interpreter_source: "ollama"  # openai | ollama | cesnet
interpreter_model: "qwen3:1.7b"
action_limit: 30
```

You can switch targets by editing the `default_target` value or adding additional entries under `targets` for your own machines.

The `llm_providers` section declares connection details for each LLM backend. Providers of kind `openai` use the OpenAI-compatible API, while providers of kind `ollama` route to a local Ollama daemon (either via `host`/`port` or a `base_url`).

**Key rules**
- Any source set to `openai` requires an API key (e.g., `OPENAI_API_KEY`, `CESNET_API_KEY`) and optionally a `base_url` declared under `llm_providers`.
- Any source set to `ollama` must have a provider entry with connection details (host/port or `base_url`).
- Provider names (`openai`, `ollama`, `cesnet`, etc.) are referenced by `planner_source`, `summarizer_source`, and `interpreter_source`.

### CESNET OpenAI-Compatible Endpoint

If you have access to the CESNET deployment you can reuse the OpenAI client without modifying the code by exporting the following variables:

```bash
export CESNET_API_KEY="<your token>"
export CESNET_BASE_URL="https://chat.ai.e-infra.cz/api"   # optional; default when CESNET_API_KEY is set
# (optional) keep legacy tools happy
export OPENAI_API_KEY="${CESNET_API_KEY}"
```

- Models default to `qwen3-coder` unless overridden in the YAML config.
- The agent automatically points the OpenAI SDK at the CESNET base URL when `CESNET_API_KEY` is present.

Example curl call:

```bash
curl https://chat.ai.e-infra.cz/api/v1/chat/completions \
  -H "Authorization: Bearer ${CESNET_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3-coder",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello from CESNET!"}
        ]
      }'
```

In your ARACNE YAML, set any role to the `cesnet` provider (for example `planner_source: "cesnet"`) and choose a compatible model such as `qwen3-coder`.

## Usage

Run from the repository root using the wrapper:

```bash
python aracne.py -h
```

```
usage: aracne.py [-h] [-e ENV] [-s] [-g GOAL] [-o OLLAMA_IP] [-c CONFIG]

Pentesting Agent CLI

optional arguments:
  -h, --help            Show help and exit
  -e ENV, --env ENV     Path to .env (default: .env)
  -s, --summarize       Enable on-the-fly summarization
  -g GOAL, --goal GOAL  Override goal from CLI
  -o OLLAMA_IP, --ollama-ip OLLAMA_IP
                        Override Ollama host/IP
  -c CONFIG, --config CONFIG
                        Path to YAML config (default: agent/config/AttackConfig.yaml)
```

CLI overrides take precedence over YAML defaults. Example:

```bash
python aracne.py \
  -e agent/.env-local \
  -c agent/config/AttackConfig_local.yaml \
  -g "Enumerate services" \
  -o 10.0.0.50 \
  -s
```

### Switching Models

- To use OpenAI for planner/summarizer: set `*_source: "openai"` and corresponding `*_model`.
- To run entirely on Ollama: set `planner_source: "ollama"`, `summarizer_source: "ollama"`, `interpreter_source: "ollama"` (or `"local"`) and supply models that exist locally (e.g., `qwen3:1.7b`).

### Docker

Build the images once:

```bash
docker compose build
```

Run the ARACNE CLI inside its container:

```bash
docker compose up -d aracne
docker exec -it aracne python aracne.py -e agent/.env -c agent/config/AttackConfig.yaml
```

#### Optional: Launch a local SSH playground

If you need a disposable SSH target you can start the companion container:

```bash
docker compose up -d ssh-target
```

The container is based on Ubuntu and exposes SSH on the host at `127.0.0.1:2222` with the following credentials:

| Field        | Value        |
|--------------|--------------|
| Username     | `root`       |
| Password     | `y54325342`  |
| Host         | `127.0.0.1`  |
| Port         | `2222`       |

To target the playground container, update your YAML or pass CLI overrides, e.g.:

```yaml
ssh_host: "127.0.0.1"
ssh_port: 2222
ssh_user: "root"
ssh_password: "y54325342"
```

Stop the playground when you are done:

```bash
docker compose stop ssh-target
```

Mount or copy configuration/env files as needed by editing `docker-compose.yml`.

## Goals

If no goal is specified, ARACNE prompts you with a list of predefined pentesting objectives (privilege escalation, detecting honeypots, exfiltration, etc.). Provide a goal via config or CLI to skip the prompt.

## Logs

Logs live under `logs/` and are rebuilt on each run:

| File              | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `agent.log`       | Colorized simulation of the remote shell (commands + output).     |
| `context.log`     | Sequenced LLM call trace with prompts, responses, summaries, etc. |

`context.log` uses ANSI colors, section dividers, and indenting for readability.

## Troubleshooting

- **Missing model**: Ensure `planner_model`/`interpreter_model` exists locally (`ollama pull <model>`).
- **OpenAI errors**: Verify `OPENAI_API_KEY` and model names match your subscription.
- **SSH failures**: Confirm `.env` has `SSH_HOST/SSH_USER/SSH_PASSWORD` values and the target is reachable.
- **Action limit**: Increase `action_limit` for longer engagements.

## License

GPL-2.0. See [LICENSE](LICENSE) for details.

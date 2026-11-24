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

Populate `agent/.env` with `CESNET_API_KEY` for the default CESNET models. SSH target profiles are configured in `agent/config/AttackConfig.yaml`; `OPENAI_API_KEY` is only needed when a role uses OpenAI.

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
    base_url: https://llm.ai.e-infra.cz/v1
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
planner_source: "cesnet"     # openai | ollama | cesnet
planner_model: "glm-5.3"
planner_temperature: 1.0      # Optional creativity control per role
summarizing: false
summarizer_source: "cesnet"  # openai | ollama | cesnet
summarizer_model: "glm-5.3"
summarizer_temperature: 1.0
interpreter_source: "cesnet"  # openai | ollama | cesnet
interpreter_model: "glm-5.3"
interpreter_temperature: 1.0
action_limit: 30
```

You can switch targets by editing the `default_target` value or adding additional entries under `targets` for your own machines.

The `llm_providers` section declares connection details for each LLM backend. Providers of kind `openai` use the OpenAI-compatible API, while providers of kind `ollama` route to a local Ollama daemon (either via `host`/`port` or a `base_url`).

**Key rules**
- Any source set to `openai` requires an API key (e.g., `OPENAI_API_KEY`, `CESNET_API_KEY`) and optionally a `base_url` declared under `llm_providers`.
- Any source set to `ollama` must have a provider entry with connection details (host/port or `base_url`).
- Provider names (`openai`, `ollama`, `cesnet`, etc.) are referenced by `planner_source`, `summarizer_source`, and `interpreter_source`.
- Per-role `*_temperature` values (0–2) are forwarded directly to the respective LLM API (`temperature` for OpenAI-compatible endpoints, `options.temperature` for Ollama).

### CESNET OpenAI-Compatible Endpoint

If you have access to the e-INFRA deployment, create an API key in the Chat AI account settings and export it:

```bash
export CESNET_API_KEY="<your token>"
```

- Set `planner_source` or `interpreter_source` to `cesnet` in YAML. ARACNE uses `https://llm.ai.e-infra.cz/v1` by default and accepts the old Chat AI URL in older configs by redirecting it to the current endpoint.
- Choose a model listed for your key. Stable aliases such as `glm`, `mini`, and `coder` are available; the exact model list can change.

Example curl call:

```bash
curl https://llm.ai.e-infra.cz/v1/chat/completions \
  -H "Authorization: Bearer ${CESNET_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "glm",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello from CESNET!"}
        ]
      }'
```

List models available to your key with `curl -H "Authorization: Bearer ${CESNET_API_KEY}" https://llm.ai.e-infra.cz/v1/models`.

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

Nmap scans can run for up to 30 minutes. ARACNE reports progress while waiting and captures the completed scan output. The interpreter's command is executed as generated.

Use `--interactive` (`-i`) to keep the CLI open for more goals. Press Ctrl+C while a command runs to interrupt it on the remote SSH shell and enter a new goal. After each goal finishes, enter another goal or type `quit` to exit. Each new goal starts a new run and SSH connection.
Interactive goals share session memory: recent commands and results stay available to the planner, and older or large results are condensed automatically. The full output remains in the logs. Session memory is saved after each action and works without `--summarize`.

To continue a saved session, run `python aracne.py --load-session` (or `--load-session latest`). ARACNE restores the last goal and opens interactive mode. Add `-g "new goal"` to continue the same history with a new instruction. You can also pass a session JSON path, an experiment directory, or its `context.log` to `--load-session`. Session files are stored in `logs/sessions/`; `latest` imports the newest archived experiment if no session file exists.

Commands that delete data, change system state, write files, or use shell syntax that is difficult to classify require an explicit `yes` at a terminal before they run. Declining or running without an interactive terminal stops the current goal without executing the command.
If a plan, generated command, or command output asks for human or system-owner confirmation, ARACNE asks you directly. An explicit `yes` is recorded as user approval in the planner context before work continues; any other answer stops the current goal.

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

Logs live under `logs/`. The active `agent.log` and `context.log` are rebuilt on each run; experiment archives and session snapshots persist:

| File              | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `agent.log`       | Colorized simulation of the remote shell (commands + output).     |
| `context.log`     | Sequenced LLM call trace with prompts, responses, summaries, etc. |
| `sessions/*.json` | Saved interactive memory for `--load-session`.                   |

`context.log` uses ANSI colors, section dividers, and indenting for readability.

## Troubleshooting

- **Missing model**: Ensure `planner_model`/`interpreter_model` exists locally (`ollama pull <model>`).
- **OpenAI errors**: Verify `OPENAI_API_KEY` and model names match your subscription.
- **SSH failures**: Confirm `.env` has `SSH_HOST/SSH_USER/SSH_PASSWORD` values and the target is reachable.
- **Action limit**: Increase `action_limit` for longer engagements.

## License

GPL-2.0. See [LICENSE](LICENSE) for details.

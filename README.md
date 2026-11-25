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
cp env_EXAMPLE .env
```

Populate `.env` with `BOT1_SSH_PASSWORD` for the default SSH target and the API key required by each selected model provider (`OPENAI_API_KEY`, `GROQ_API_KEY`, or `CESNET_API_KEY`). SSH target profiles and model selections are configured in `agent/config/AttackConfig.yaml`.

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
  groq:
    kind: openai
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
targets:                     # Named SSH target profiles
  bot1:
    host: 127.0.0.1
    port: 22
    user: pedro
    password_env: BOT1_SSH_PASSWORD
  ubuntu:
    host: localhost
    port: 2222
    user: root
    password: 123456
  shellm:
    host: localhost
    port: 22
    user: admin
    password: admin
default_target: bot1          # Which profile to load by default
planner_source: "groq"       # openai | ollama | cesnet | groq
planner_model: "openai/gpt-oss-120b"
planner_temperature: 1.0      # Optional creativity control per role
summarizing: false
summarizer_source: "openai"  # openai | ollama | cesnet | groq
summarizer_model: "gpt-6.1-sol"
summarizer_temperature: 1.0
interpreter_source: "groq"    # openai | ollama | cesnet | groq
interpreter_model: "openai/gpt-oss-120b"
interpreter_temperature: 1.0
action_limit: 50
```

You can switch targets by editing the `default_target` value or adding additional entries under `targets` for your own machines.
Use `password_env` to read a target password from the named environment variable instead of storing it in YAML. The default `bot1` profile reads `BOT1_SSH_PASSWORD` from `.env` or the process environment.

The `llm_providers` section declares connection details for each LLM backend. Providers of kind `openai` use the OpenAI-compatible API, while providers of kind `ollama` route to a local Ollama daemon (either via `host`/`port` or a `base_url`).

**Key rules**
- Any source set to `openai` requires an API key (e.g., `OPENAI_API_KEY`, `CESNET_API_KEY`) and optionally a `base_url` declared under `llm_providers`.
- Any source set to `ollama` must have a provider entry with connection details (host/port or `base_url`).
- Provider names (`openai`, `ollama`, `cesnet`, etc.) are referenced by `planner_source`, `summarizer_source`, and `interpreter_source`.
- Per-role `*_temperature` values (0–2) are forwarded directly to the respective LLM API (`temperature` for OpenAI-compatible endpoints, `options.temperature` for Ollama).

### Agent prompts

Edit the role prompts in `agent/config/plannerConfig.yml`, `agent/config/interpreterConfig.yml`, `agent/config/summarizerConfig.yml`, and `agent/config/finalSummaryConfig.yml`. Session-memory compression uses the prompt in `agent/lib/memory.py`. These prompts apply to the selected role regardless of whether its model runs on OpenAI, Groq, CESNET, or Ollama.

The planner and interpreter prompts include the user's statement that ARACNE may run commands needed for the stated assessment and that the authorization is verified and signed. ARACNE treats this as a user assertion; it does not verify signatures or ownership. Target scope still comes from the goal, and the runtime asks before deletion operations.

To use Groq, put `GROQ_API_KEY=...` in the ignored root `.env` file (or export it in the process environment). Set any role's `*_source` to `groq` and its matching `*_model` to `openai/gpt-oss-120b`. The default CLI reads the root `.env`; use `-e` to select a different file. The provider entry uses `kind: openai`, and the model is selected per role rather than in `llm_providers`.

### OpenAI model comparison

In the user-provided comparison dated November 25, 2026, GPT-6.1 Sol (`gpt-6.1-sol`) is the first choice for balancing intelligence and cost. For large-scale experiments, the comparison favors models with a larger reported daily token pool over a more capable model when throughput matters most.

| API model ID | Input $/1M | Output $/1M | Reported free tokens/day* | Best for |
| --- | ---: | ---: | ---: | --- |
| `gpt-6-astra` | $10 | $50 | 1M | Maximum intelligence |
| `gpt-6.1-sol` | $2 | $10 | 1M | Intelligence/cost balance |
| `gpt-5.6-sol` | $4 | $20 | 1M | Complex reasoning |
| `gpt-5.6-terra` | $2 | $12 | 10M | Large-scale experiments |
| `gpt-5.6-luna` | $0.20 | $1.20 | 10M | Cheap general processing |
| `gpt-6-luna` | $0.10 | $0.50 | 1M | Extremely cheap processing |

*The daily token amounts come from the supplied comparison; they are not a universal OpenAI API entitlement. Check the allowance for your account or offer. Prices are standard text-token rates; check [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) for current rates and other processing tiers.

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

#### CESNET model ranking

The following user-provided ranking is a qualitative guide for complex reasoning, programming, tool use, and autonomous agents as of November 25, 2026. Model availability depends on your CESNET account; check `/v1/models` before selecting an API model ID.
The ranked IDs are listed as commented choices for each role in `agent/config/AttackConfig.yaml`; select one model per role and set its source to `cesnet`.

| Rank | API model ID | Intelligence tier | Main strength |
| ---: | --- | --- | --- |
| 1 | `kimi-k3.1-pro` | Frontier | Complex agents, planning, coding |
| 2 | `glm-5.3` | Frontier | Reasoning, coding, tool use |
| 3 | `deepseek-v4.1-flash-thinking` | Frontier | Reasoning, mathematics |
| 4 | `kimi-k3` | Very high | Complex reasoning, agents |
| 5 | `qwen3.8-flash-next` | Very high | Agentic coding, efficiency |
| 6 | `deepseek-v4.1-flash` | Very high | General tasks, speed |
| 7 | `glm-5` | Very high | Reasoning, programming |
| 8 | `glm-5.3-flash` | High | Fast inference |
| 9 | `qwen3.8-27b` | High | Coding, efficiency |
| 10 | `qwen3.5-int4` | High | General reasoning |
| 11 | `gpt-oss-120b` | High | Reasoning, reproducibility |
| 12 | `mistral-medium-3.5` | High | General-purpose tasks |
| — | `gemma4` | Undetermined | Depends on parameter count |

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
At a goal prompt, use Up and Down to revisit goals you entered earlier. ARACNE saves accepted goals across runs in `logs/input_history` with owner-only permissions. Approval answers and `quit` are not saved. Arrow-key history requires an interactive terminal with Python readline support.
Interactive goals share session memory: recent commands and results stay available to the planner, and older or large results are condensed automatically. The full output remains in the logs. Session memory is saved after each action and works without `--summarize`.

To continue a saved session, run `python aracne.py --load-session` (or `--load-session latest`). ARACNE loads the history and waits for a new goal before contacting the LLM. Add `-g "new goal"` to provide that goal immediately. You can also pass a session JSON path, an experiment directory, or its `context.log` to `--load-session`. Session files are stored in `logs/sessions/`; `latest` imports the newest archived experiment if no session file exists.

ARACNE asks for approval before recognizable deletion operations, including `rm`, `find -delete`, `git clean`, and HTTP `DELETE` requests. Browsing and fetching with `curl`, `wget`, `lynx`, `w3m`, or similar tools runs without an approval prompt. Other commands, redirects, shell loops, and unknown executables also run without approval by default. Edit [agent/config/commandApproval.yaml](agent/config/commandApproval.yaml) to add or remove executable names in `always_ask`; ARACNE reloads the file before each command. Set `ask_unknown_commands: true` only if you want unknown executables reviewed. Declining a deletion returns to the new-goal prompt.
If a plan, generated command, or command output specifically asks for human or system-owner confirmation before deletion, ARACNE asks you directly. An explicit `yes` is recorded as user approval before work continues; any other answer stops the current goal.

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

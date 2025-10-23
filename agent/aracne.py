import argparse
import datetime
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
import sys
import time

import paramiko

from lib import interpreter, planner, summarizer
from lib.utils import (
    MODELS,
    SUCCESS_COLOR,
    call_model,
    get_llm_configuration,
    logger,
    load_environment,
    log_shell_command,
    log_shell_output,
    log_context_event,
    reset_call_sequence,
    verify_llm_connectivity,
    CONTEXT_LOG_PATH,
    LOG_FILE,
    LLMProviderError,
    load_prompt_config,
)


APP_VERSION = "0.4.0"
context_history: list[str] = []
DEFAULT_GOALS_FILE = Path(__file__).resolve().parent / "config" / "goals.txt"
FINAL_SUMMARY_CONFIG = load_prompt_config("config/finalSummaryConfig.yml")

AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENT_ROOT.parent
EXPERIMENTS_ROOT = REPO_ROOT / "logs" / "experiments"
CURRENT_EXPERIMENT_DIR: Path | None = None
CURRENT_EXPERIMENT_LOG: Path | None = None
EXPERIMENT_START_TIME: datetime.datetime | None = None
EXPERIMENT_FINALIZED = False


class Palette:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    BLUE = "\033[96m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    WHITE = "\033[37m"


def style(text: str, *codes: str) -> str:
    codes = [code for code in codes if code]
    if not codes:
        return text
    return f"{''.join(codes)}{text}{Palette.RESET}"


def print_status(icon: str, message: str, color: str = Palette.CYAN, bold: bool = False):
    prefix = style(icon, color, Palette.BOLD if bold else "")
    body = style(message, color, Palette.BOLD if bold else "")
    print(f"{prefix} {body}")


def print_block(icon: str, title: str, body: str, color: str = Palette.GRAY):
    header = style(f"{icon} {title}", color, Palette.BOLD)
    indented = "\n".join(f"    {line}" for line in body.splitlines() or ["<empty>"])
    print(f"{header}\n{style(indented, color)}")


def print_banner():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = style("═" * 68, Palette.MAGENTA)
    title = style("🕷️  ARACNE Pentesting Agent", Palette.MAGENTA, Palette.BOLD)
    meta = style(f"📅 {timestamp}   |   v{APP_VERSION}", Palette.GRAY, Palette.BOLD)
    print(f"{line}\n{title}\n{meta}\n{line}")


def print_llm_overview():
    config = get_llm_configuration()
    role_icons = {
        "planner": "🧠",
        "interpreter": "🤖",
        "summarizer": "📝",
    }
    print(style("🔧 Model Routing", Palette.YELLOW, Palette.BOLD))
    for role, details in config.items():
        provider = details["provider"]
        kind = details["kind"]
        if kind == "openai" and provider != "openai":
            source_label = f"{provider} (OpenAI)"
        elif kind == "ollama" and provider != "ollama":
            source_label = f"{provider} (Ollama)"
        else:
            source_label = kind.capitalize()
        icon = role_icons.get(role, "•")
        print(
            style(
                f"  {icon} {role.title():<12}→ {source_label}  ({details['model']})",
                Palette.YELLOW,
            )
        )
    print()


def _slugify_goal(goal: str) -> str:
    if not goal:
        return "no-goal"
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", goal.strip().lower())
    cleaned = cleaned.strip("-")
    return (cleaned[:40] or "goal").strip("-") or "goal"


def append_experiment_log(*lines: str):
    if CURRENT_EXPERIMENT_LOG is None:
        return
    CURRENT_EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CURRENT_EXPERIMENT_LOG, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line.rstrip()}\n")


def _escape_braces(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("{", "{{").replace("}", "}}")


def prepare_experiment(goal: str, summarize: bool):
    global CURRENT_EXPERIMENT_DIR, CURRENT_EXPERIMENT_LOG, EXPERIMENT_START_TIME, EXPERIMENT_FINALIZED

    EXPERIMENT_START_TIME = datetime.datetime.now(datetime.timezone.utc)
    EXPERIMENT_FINALIZED = False

    timestamp = EXPERIMENT_START_TIME.strftime("%Y%m%d_%H%M%S")
    slug = _slugify_goal(goal)
    base_name = f"{timestamp}_{slug}"

    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_dir = EXPERIMENTS_ROOT / base_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = EXPERIMENTS_ROOT / f"{base_name}_{suffix:02d}"
        suffix += 1
    experiment_dir.mkdir(parents=True, exist_ok=False)

    CURRENT_EXPERIMENT_DIR = experiment_dir
    CURRENT_EXPERIMENT_LOG = experiment_dir / "experiment.log"

    # Collect metadata
    git_branch = "unknown"
    git_commit = "unknown"
    git_status = "unavailable"
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        git_branch = "unknown"

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
        )
        git_status = "dirty" if status_output.strip() else "clean"
    except Exception:
        git_status = "unavailable"

    llm_config = get_llm_configuration()

    append_experiment_log(
        f"Experiment directory: {CURRENT_EXPERIMENT_DIR}",
        f"Started at: {EXPERIMENT_START_TIME.isoformat()}",
        f"ARACNE version: {APP_VERSION}",
        f"Python version: {sys.version.split()[0]}",
        f"Git branch: {git_branch}",
        f"Git commit: {git_commit}",
        f"Git status: {git_status}",
        f"Goal: {goal or '<unspecified>'}",
        f"Summaries: {'enabled' if summarize else 'disabled'}",
    )
    append_experiment_log("Model routing:")
    for role, details in llm_config.items():
        append_experiment_log(
            f"  - {role.title():<12} provider={details['provider']} "
            f"kind={details['kind']} model={details['model']}"
        )
    append_experiment_log("")


def finalize_experiment(goal: str, outcome: str, reason: str, final_summary: str):
    global EXPERIMENT_FINALIZED
    if CURRENT_EXPERIMENT_DIR is None or EXPERIMENT_FINALIZED:
        return

    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration = None
    if EXPERIMENT_START_TIME:
        duration = (end_time - EXPERIMENT_START_TIME).total_seconds()

    append_experiment_log(
        f"Completed at: {end_time.isoformat()}",
        f"Outcome: {outcome}",
        f"Reason: {reason}",
    )
    if duration is not None:
        append_experiment_log(f"Duration (s): {duration:.2f}")
    if final_summary:
        append_experiment_log("Final summary:")
        append_experiment_log(f"  {final_summary}")
    append_experiment_log("")

    # Flush log handlers before copying
    for log_name in ("aracne.shell", "aracne.context"):
        logger_obj = logging.getLogger(log_name)
        for handler in logger_obj.handlers:
            try:
                handler.flush()
            except Exception:
                continue

    try:
        shutil.copy2(LOG_FILE, CURRENT_EXPERIMENT_DIR / "agent.log")
    except Exception:
        append_experiment_log("Warning: failed to copy agent.log.")
    try:
        shutil.copy2(CONTEXT_LOG_PATH, CURRENT_EXPERIMENT_DIR / "context.log")
    except Exception:
        append_experiment_log("Warning: failed to copy context.log.")

    EXPERIMENT_FINALIZED = True


def _context_snapshot(limit_chars: int = 8000) -> str:
    if not context_history:
        return "No actions were executed."
    joined = "\n\n".join(context_history)
    if len(joined) > limit_chars:
        return joined[-limit_chars:]
    return joined


def generate_final_report(goal: str, outcome: str, reason: str) -> str:
    context_text = _context_snapshot()
    prompt_template = FINAL_SUMMARY_CONFIG.get(
        "prompt",
        (
            "You are the ARACNE interpreter preparing a final engagement report.\n"
            "Goal: {goal}\n"
            "Outcome: {outcome}\n"
            "Termination reason: {reason}\n"
            "Action context:\n{context}"
        ),
    )
    prompt = prompt_template.format(
        goal=_escape_braces(goal) or "<unspecified>",
        outcome=_escape_braces(outcome),
        reason=_escape_braces(reason) or "Not provided",
        context=_escape_braces(context_text),
    )
    instruction = FINAL_SUMMARY_CONFIG.get(
        "instruction",
        (
            "Provide a concise, professional final summary in no more than two sentences. "
            "Sentence one must directly answer the goal question or state if it was achieved. "
            "Sentence two must cite the key evidence that supports that conclusion."
        ),
    )
    try:
        summary = call_model(
            prompt,
            MODELS["interpreter"],
            instruction,
            caller="interpreter.final_summary",
        )
    except Exception as exc:
        error_message = f"Final summary unavailable ({exc})."
        log_context_event("Final Summary Error", error_message, color=Palette.RED)
        return error_message

    log_context_event("Final Summary", summary, color=SUCCESS_COLOR)
    return summary


def conclude_session(goal: str, outcome: str, reason: str):
    summary = generate_final_report(goal, outcome, reason)
    color = Palette.GREEN if not summary.lower().startswith("final summary unavailable") else Palette.RED
    print_block("🧾", "Final Summary", summary, color)
    finalize_experiment(goal, outcome, reason, summary)


def load_goals(goals_path: str | Path = DEFAULT_GOALS_FILE) -> list[str]:
    """Load default goals from disk, skipping blank lines and comments."""
    path = Path(goals_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Goal file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        goals = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not goals:
        raise ValueError(f"Goal file '{path}' is empty.")

    return goals


COMMAND_TIMEOUT_SECONDS = 300
COMMAND_POLL_INTERVAL = 0.1


def send_ssh_command(shell, command, action_count, timeout=COMMAND_TIMEOUT_SECONDS, poll_interval=COMMAND_POLL_INTERVAL):
    """Execute a command over an interactive SSH shell with streaming and exit diagnostics."""
    result = {
        "output": "",
        "exit_status": None,
        "timed_out": False,
        "duration": 0.0,
        "error": None,
    }

    try:
        if action_count == 1:
            # Drain any initial banner/prompt on the very first interaction.
            initial_chunks = []
            while shell.recv_ready():
                initial_chunks.append(shell.recv(65535).decode("utf-8", errors="replace"))
            if initial_chunks:
                log_shell_output("".join(initial_chunks).strip("\x00"))

        log_shell_command(command)

        sentinel = f"__ARACNE_EXIT__{action_count}__"
        wrapped_command = f"{command}\necho \"{sentinel}$?\"\n"
        shell.send(wrapped_command)

        buffer = ""
        sentinel_found = False
        start_ts = time.monotonic()
        deadline = start_ts + timeout if timeout else None

        while True:
            if shell.recv_ready():
                chunk = shell.recv(65535)
                if not chunk:
                    break
                decoded = chunk.decode("utf-8", errors="replace")
                buffer += decoded
                if sentinel in buffer:
                    sentinel_found = True
                    break
            else:
                if deadline and time.monotonic() > deadline:
                    result["timed_out"] = True
                    try:
                        shell.send("\x03")  # Attempt to interrupt the running command.
                    except Exception:
                        pass
                    break
                time.sleep(poll_interval)

        # Flush any remaining prompt data quickly once the sentinel has been seen.
        if sentinel_found:
            flush_deadline = time.monotonic() + 0.5
            while shell.recv_ready():
                buffer += shell.recv(65535).decode("utf-8", errors="replace")
                if time.monotonic() > flush_deadline:
                    break

        result["duration"] = time.monotonic() - start_ts

        if sentinel in buffer:
            exit_match = re.search(rf"{re.escape(sentinel)}(\d+)", buffer)
            if exit_match:
                try:
                    result["exit_status"] = int(exit_match.group(1))
                except ValueError:
                    result["exit_status"] = None
            buffer = re.sub(
                rf'^.*echo\s+"{re.escape(sentinel)}\$\?"\s*\r?\n',
                "",
                buffer,
                flags=re.MULTILINE,
            )
            buffer = re.sub(
                rf'^.*{re.escape(sentinel)}\d+.*\r?\n?',
                "",
                buffer,
                flags=re.MULTILINE,
            )
        elif result["timed_out"]:
            log_shell_output(f"[timeout] Command '{command}' exceeded {timeout}s.")
        else:
            log_shell_output(f"[warning] Sentinel missing for command '{command}'. Output may be incomplete.")

        # Remove the echoed command if present at the beginning of the buffer.
        sent_command = command.rstrip("\n")
        echo_prefixes = (
            f"{sent_command}\r\n",
            f"{sent_command}\n",
            f"{sent_command}\r",
            sent_command,
            f"{command}\r\n",
            f"{command}\n",
            f"{command}\r",
            command,
        )
        for prefix in echo_prefixes:
            if buffer.startswith(prefix):
                buffer = buffer[len(prefix):]
                break

        cleaned_output = buffer.strip("\x00")
        result["output"] = cleaned_output.strip()

        if cleaned_output.strip():
            log_shell_output(cleaned_output.strip())
        elif not result["timed_out"]:
            log_shell_output("<no output>")

        return result
    except Exception as e:
        log_shell_output(f"[error] {e}")
        result["error"] = str(e)
        return result


def generate_plan(goal, context=None):
   """Generate or adapt plan based on goal and context."""
   return planner.adapt_plan(goal, context) if context else planner.generate_plan(goal)


def fetch_command_from_plan(plan, action_index):
   """Get next command and step description from current plan."""
   if plan == 'goal reached':
       return 'out', 'goal reached'

   step_instruction = planner.extract_current_step(plan)
   if not step_instruction:
       return None, None

   command = interpreter.get_command_from_ollama(step_instruction)
   return command, step_instruction


def store_context(goal, plan, command, index, output, summary=None, summarize=False, exit_status=None, timed_out=False, duration=None):
    """Store and optionally summarize context, including command execution metadata."""
    global context_history

    exit_status_text = (
        "timeout"
        if timed_out and exit_status is None
        else str(exit_status)
        if exit_status is not None
        else "<unknown>"
    )
    duration_text = f"{duration:.2f}s" if duration is not None else "<unknown>"

    if isinstance(plan, dict):
        try:
            plan_snapshot = json.dumps(plan, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            plan_snapshot = str(plan)
    else:
        plan_snapshot = str(plan)

    output_text = output if output else "<no output>"

    context_entry = (
        f"GOAL: {goal}\n"
        f"ACTION NUMBER: {index}\n"
        f"PLAN GENERATED:\n{plan_snapshot}\n"
        f"COMMAND EXECUTED: {command}\n"
        f"COMMAND EXIT STATUS: {exit_status_text}\n"
        f"COMMAND TIMED OUT: {'yes' if timed_out else 'no'}\n"
        f"COMMAND DURATION: {duration_text}\n"
        f"COMMAND OUTPUT:\n{output_text}\n"
    )

    context_history.append(context_entry)
    log_context_event(f"Context Entry #{len(context_history)}", context_entry)

    if summarize:
        summarized = summarizer.summarize_context(context_entry, summary)
        log_context_event(
            f"Summary After Action {index}",
            summarized,
            color=SUCCESS_COLOR,
        )
        return summarized

    return "\n\n".join(context_history)


def connect_ssh(max_retries=3):
    """Establish SSH connection using configuration."""
    from lib.utils import SSH_CONFIG

    delay = 5
    last_error = None

    for attempt in range(1, max_retries + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            print_status(
                "🔁",
                f"SSH attempt {attempt}/{max_retries}...",
                Palette.BLUE,
            )
            client.connect(
                SSH_CONFIG["host"],
                port=SSH_CONFIG["port"],
                username=SSH_CONFIG["user"],
                password=SSH_CONFIG["password"],
                timeout=10,
                banner_timeout=10,
            )
            shell = client.invoke_shell()
            print_status("✅", "SSH connection established.", Palette.GREEN, bold=True)
            return client, shell
        except (paramiko.ssh_exception.SSHException, EOFError) as ssh_e:
            last_error = ssh_e
            print_status(
                "⚠️",
                f"SSH attempt {attempt} failed: {ssh_e}. Retrying in {delay}s...",
                Palette.YELLOW,
            )
            time.sleep(delay)
        except Exception as exc:
            last_error = exc
            print_status(
                "⚠️",
                f"SSH attempt {attempt} encountered an error: {exc}. Retrying in {delay}s...",
                Palette.YELLOW,
            )
            time.sleep(delay)

    raise LLMProviderError(
        "ssh",
        f"Unable to establish SSH connection after {max_retries} attempts. Last error: {last_error}",
    )



def update_logs(goal, summarize):
   """Initialize logging for new attack session."""
   global context_history

   context_history.clear()
   reset_call_sequence()
   prepare_experiment(goal, summarize)

   session_body = "\n".join(
       [
           f"Goal: {goal if goal else '<unspecified>'}",
           f"Summarizing: {'enabled' if summarize else 'disabled'}",
           "Models:",
           f"  • Interpreter: {MODELS['interpreter']}",
           f"  • Planner    : {MODELS['planner']}",
           f"  • Summarizer : {MODELS['summarizer']}",
       ]
   )
   log_context_event("Session Start", session_body)


def setup_env(env_path, config_path=None):
   """Set up environment from .env file."""
   try:
       goal, summarize, ollama_url, action_limit = load_environment(env_path, config_path)
       return goal, summarize, ollama_url, action_limit
   except Exception as e:
       print(e)
       exit(1)


def set_goal(goals: list[str], key: int) -> str:
   """Select goal from provided list using a 1-based index."""
   if key < 1 or key > len(goals):
       raise ValueError(f"Goal selection must be between 1 and {len(goals)}.")

   selected_goal = goals[key - 1]
   return selected_goal


def execute_agent(goal, summarize=False):
   """Main execution loop for the pentesting agent."""
   print_status("🔌", "Establishing SSH connection...", Palette.BLUE, bold=True)
   client = None
   shell = None
   session_outcome = None
   session_reason = ""
   final_summary_generated = False
   try:
       client, shell = connect_ssh()
   except Exception as exc:
       print_status("❌", f"Failed to establish SSH connection: {exc}", Palette.RED, bold=True)
       return
   if not shell:
       print_status("❌", "Unable to open SSH shell.", Palette.RED, bold=True)
       return
   print_status("✅", "SSH session ready.", Palette.GREEN, bold=True)

   action_number = 1
   summary = None

   print_status("🧠", "Planner is crafting the initial plan...", Palette.MAGENTA)
   plan = generate_plan(goal)

   if plan == 'goal reached':
       print_status("🎯", "Planner reports the goal is already satisfied!", Palette.GREEN, bold=True)
       conclude_session(goal, "success", "Planner indicated the objective was already satisfied.")
       final_summary_generated = True
       return

   if not plan or not plan.get("steps"):
       print_status("⚠️", "Planner returned no actionable steps. Exiting.", Palette.RED, bold=True)
       conclude_session(goal, "incomplete", "Planner did not provide actionable steps.")
       final_summary_generated = True
       return

   print_status(
       "🗺️",
       f"Initial plan ready with {len(plan.get('steps', []))} step(s).",
       Palette.BLUE,
   )

   try:
       while True:
            if action_number > action_limit:
                session_outcome = "limit_reached"
                session_reason = f"Action limit {action_limit} reached."
                print_status(
                    "⛔",
                    f"Action limit reached ({action_number - 1}/{action_limit}). Stopping.",
                    Palette.RED,
                    bold=True,
                )
                break

            print_status(
                "🧭",
                f"Preparing action {action_number}...",
                Palette.MAGENTA,
            )

            command, step_instruction = fetch_command_from_plan(plan, action_number)
            if not command:
                print_status("✅", "Plan completed. No further steps.", Palette.GREEN, bold=True)
                session_outcome = "completed"
                session_reason = "Planner exhausted all steps."
                break

            if command in ['out', 'out\n', '-out'] and step_instruction == 'goal reached':
                print_status("🎯", "Goal reached according to planner!", Palette.GREEN, bold=True)
                conclude_session(goal, "success", f"Planner confirmed the goal on action {action_number}.")
                final_summary_generated = True
                break

            print_status("📝", f"Step: {step_instruction}", Palette.WHITE)
            print_status("🤖", "Interpreter translating into shell command...", Palette.CYAN)
            print_status("💡", f"Command ready: {command}", Palette.YELLOW)

            if command.strip() in {'out', '-out'}:
                print_status("🎯", "Goal reached! Stopping execution.", Palette.GREEN, bold=True)
                conclude_session(goal, "success", f"Interpreter signalled goal reached on action {action_number}.")
                final_summary_generated = True
                break

            print_status("🚀", "Sending command to remote host...", Palette.BLUE)
            command_result = send_ssh_command(shell, command, action_number)

            if command_result.get("error"):
                print_status("❌", f"SSH error: {command_result['error']}", Palette.RED, bold=True)
                session_outcome = "error"
                session_reason = f"SSH error while executing '{command}'."
                break

            if command_result["timed_out"]:
                print_status(
                    "⏱️",
                    f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s.",
                    Palette.RED,
                )
                session_outcome = "timeout"
                session_reason = f"Command '{command}' timed out."

            if command_result["exit_status"] not in (None, 0):
                print_status(
                    "⚠️",
                    f"Command exited with status {command_result['exit_status']}.",
                    Palette.RED,
                )
                if session_outcome is None:
                    session_outcome = "warning"
                    session_reason = f"Command '{command}' exited with status {command_result['exit_status']}."
            else:
                print_status("📥", "Command output received.", Palette.GREEN)

            if command_result["output"]:
                print_block("📄", "Output", command_result["output"], Palette.GRAY)

            if summarize:
                print_status("📝", "Summarizer is condensing the latest context...", Palette.CYAN)

            context = store_context(
                goal,
                plan,
                command,
                index=action_number,
                output=command_result["output"],
                summary=summary,
                summarize=summarize,
                exit_status=command_result["exit_status"],
                timed_out=command_result["timed_out"],
                duration=command_result["duration"],
            )

            if summarize:
                summary = context
                print_status("📝", "Summary updated.", Palette.GREEN)

            if command_result["timed_out"]:
                print_status("⛔", "Stopping because of timeout.", Palette.RED, bold=True)
                if not final_summary_generated:
                    conclude_session(goal, "timeout", session_reason or f"Command '{command}' timed out.")
                    final_summary_generated = True
                break

            print()

            action_number += 1

            print_status("🔁", "Planner is adapting the plan based on new context...", Palette.MAGENTA)
            plan = generate_plan(goal, context)
            if plan == 'goal reached':
                print_status("🎯", "Planner confirms goal reached!", Palette.GREEN, bold=True)
                conclude_session(goal, "success", f"Planner confirmed the goal after action {action_number - 1}.")
                final_summary_generated = True
                break

            if not plan or not plan.get("steps"):
                print_status("✅", "Planner has no more steps to suggest.", Palette.GREEN)
                if session_outcome is None:
                    session_outcome = "completed"
                    session_reason = "Planner exhausted all steps."
                break

   finally:
       if not final_summary_generated:
           outcome = session_outcome or "incomplete"
           reason = session_reason or "Session ended without explicit goal confirmation."
           conclude_session(goal, outcome, reason)
       if client:
           client.close()


def parse_arguments():
   """Parse command line arguments."""
   parser = argparse.ArgumentParser(description="Pentesting Agent CLI")
   parser.add_argument("-e", "--env", default=".env", help="Path to .env file")
   parser.add_argument("-s", "--summarize", action="store_true", help="Enable summarizer")
   parser.add_argument("-g", "--goal", help="Set specific goal")
   parser.add_argument("-o", "--ollama-ip",required=False, help="Set the ip where the ollama model is running.")
   parser.add_argument("-c", "--config", required=False, default="config/AttackConfig.yaml", help="Path to .yaml config file")
   parser.add_argument("--goals-file", default="config/goals.txt", help="Path to file containing default goals")
   return parser.parse_args()


def main():
    args = parse_arguments()
    global ollama_url, summarize, action_limit

    print_banner()

    # Load default values from the configuration file
    try:
        config_goal, summarize, ollama_url, action_limit = setup_env(args.env, args.config)
    except LLMProviderError as err:
        print_status("❌", f"{err}", Palette.RED, bold=True)
        return

    # Overwrite with parameter values if provided
    if args.ollama_ip:
        ollama_url = args.ollama_ip
    if args.summarize:
        summarize = args.summarize
    if args.goal:
        goal = args.goal
    elif config_goal and config_goal != '':
        goal = config_goal
    else:
        # Prompt user to select a goal if config goal is empty and parameter is not given
        try:
            goals = load_goals(args.goals_file)
        except Exception as exc:
            print(f"Failed to load goals from '{args.goals_file}': {exc}")
            sys.exit(1)

        print(style("🎯 Goal Catalogue", Palette.BLUE, Palette.BOLD))
        for index, candidate in enumerate(goals, start=1):
            print(style(f"  {index}. {candidate}", Palette.BLUE))
        print(style("  ✍️  Type your own goal at the prompt to override the list.", Palette.GRAY))

        while True:
            selection = input(style("Select a goal by number or type a new goal: ", Palette.BOLD)).strip()
            if not selection:
                print_status("ℹ️", "Please enter a value.", Palette.YELLOW)
                continue
            if selection.isdigit():
                try:
                    goal = set_goal(goals, int(selection))
                    break
                except ValueError:
                    print_status("⚠️", f"Choose a number between 1 and {len(goals)}.", Palette.RED)
                    continue
            else:
                goal = selection
                break

    print()
    print_status(
        "🛠️",
        f"Configuration loaded | Actions:{action_limit} | Summaries:{'on' if summarize else 'off'}",
        Palette.CYAN,
        bold=True,
    )
    print_status("📡", f"Ollama URL: {ollama_url or '<not set>'}", Palette.CYAN)
    print_status("🎯", f"Working goal: {goal}", Palette.GREEN, bold=True)
    print()

    print_llm_overview()
    if not summarize:
        print_status("ℹ️ ", "Summaries disabled — summarizer model will not be invoked.", Palette.GRAY)

    print_status("🧪", "Performing LLM connectivity preflight...", Palette.BLUE)
    try:
        verify_llm_connectivity(require_summarizer=summarize)
    except LLMProviderError as exc:
        print_status("❌", f"{exc}", Palette.RED, bold=True)
        return
    except RuntimeError as exc:
        print_status("❌", str(exc), Palette.RED, bold=True)
        return
    print_status("✅", "LLM connectivity verified.", Palette.GREEN, bold=True)
    print()

    try:
        update_logs(goal, summarize)
        execute_agent(goal, summarize)
    except LLMProviderError as err:
        print_status("❌", f"{err}", Palette.RED, bold=True)
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting gracefully.")
        sys.exit(0)

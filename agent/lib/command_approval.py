"""User-configurable review gate for deletion commands."""

import re
import shlex
import sys
from pathlib import Path

import yaml


POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "commandApproval.yaml"
SEPARATORS = {"&&", "||", ";", "|", "&", "\n"}
SHELL_PUNCTUATION = set(";&|()<>\n")
SHELL_CONTROL = {"if", "elif", "then", "else", "do", "while", "until"}
SHELL_HEADERS = {"for", "select", "case"}
SHELL_ENDINGS = {"fi", "done", "esac"}


def _load_policy(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    if not isinstance(policy, dict):
        raise ValueError("Policy must be a YAML mapping")
    for field in ("always_ask", "allow_without_approval"):
        value = policy.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be a list of executable names")
    if not isinstance(policy.get("ask_unknown_commands"), bool):
        raise ValueError("ask_unknown_commands must be true or false")
    return (
        {name.lower() for name in policy["always_ask"]},
        {name.lower() for name in policy["allow_without_approval"]},
        policy["ask_unknown_commands"],
    )


def _shell_tokens(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = []
    for token in lexer:
        if len(token) > 1 and set(token) <= SHELL_PUNCTUATION and token not in {
            "&&", "||", ">>", "<<", "<<<", ">&", "&>", "&>>", ">|",
        }:
            tokens.extend(re.findall(r"&&|\|\||&>>|>>|<<|>&|&>|>\||[;&|()<>\n]", token))
        else:
            tokens.append(token)
    return tokens


def _command_segments(tokens):
    segment = []
    for token in tokens:
        if token in SEPARATORS:
            if segment:
                yield segment
                segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def _executable(segment):
    for index, token in enumerate(segment):
        if token in SHELL_HEADERS:
            return None, []
        if token in SHELL_ENDINGS:
            continue
        if token in SHELL_CONTROL:
            continue
        if token in {"(", ")"} or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", token):
            continue
        if token in {"<", "<<", "<<<", ">", ">>", ">&", "&>", ">&>"}:
            continue
        if token.isdigit() and index + 1 < len(segment) and ">" in segment[index + 1]:
            continue
        return token.rsplit("/", 1)[-1].lower(), segment[index + 1:]
    return None, []


def _review(command, policy, depth=0):
    if depth > 8:
        return []
    always_ask, allowed, ask_unknown = policy
    try:
        tokens = _shell_tokens(command)
    except ValueError:
        return []
    if not tokens:
        return []

    reasons = []
    for segment in _command_segments(tokens):
        executable, arguments = _executable(segment)
        if not executable:
            continue
        if executable == "command" and arguments[:1] in (["-v"], ["-V"]):
            continue
        if executable in {"timeout", "sudo", "env", "command", "nohup", "time", "xargs"}:
            nested = next((index for index, word in enumerate(arguments) if not word.startswith("-") and not re.fullmatch(r"\d+(?:\.\d+)?[smhd]?|[A-Za-z_][A-Za-z_0-9]*=.*", word)), None)
            if nested is not None:
                reasons.extend(_review(" ".join(arguments[nested:]), policy, depth + 1))
            continue
        if executable in {"bash", "sh", "zsh"} and "-c" in arguments:
            script_index = arguments.index("-c") + 1
            if script_index < len(arguments):
                reasons.extend(_review(arguments[script_index], policy, depth + 1))
            continue
        if executable in always_ask:
            reasons.append(f"'{executable}' deletes data or is configured to require approval")
        elif executable == "find":
            if "-delete" in arguments:
                reasons.append("'find -delete' deletes files")
            for index, word in enumerate(arguments):
                if word in {"-exec", "-execdir"}:
                    reasons.extend(_review(" ".join(arguments[index + 1:]), policy, depth + 1))
        elif executable == "git" and (
            any(word in {"clean", "rm"} for word in arguments[:1])
            or "--delete" in arguments
            or (arguments[:1] == ["branch"] and any(word in {"-d", "-D", "--delete"} for word in arguments[1:]))
        ):
            reasons.append("'git' command deletes files or refs")
        elif executable in {"docker", "podman"} and (
            any(word in {"rm", "rmi", "prune"} for word in arguments)
        ):
            reasons.append(f"'{executable}' command deletes resources")
        elif executable == "kubectl" and "delete" in arguments:
            reasons.append("'kubectl delete' deletes resources")
        elif executable == "ip" and any(word in {"del", "delete", "flush"} for word in arguments):
            reasons.append("'ip' command deletes network entries")
        elif executable == "arp" and any(word in {"-d", "--delete"} for word in arguments):
            reasons.append("'arp' command deletes a neighbor entry")
        elif executable == "rsync" and any(word.startswith("--delete") for word in arguments):
            reasons.append("'rsync --delete' deletes destination files")
        elif executable == "curl" and any(
            word.upper() == "DELETE" and index > 0 and arguments[index - 1] in {"-X", "--request"}
            or word.upper() in {"--REQUEST=DELETE", "-XDELETE"}
            for index, word in enumerate(arguments)
        ):
            reasons.append("'curl' DELETE request deletes a remote resource")
        elif executable == "wget" and any(
            word.upper() == "--METHOD=DELETE"
            or word.upper() == "DELETE" and index > 0 and arguments[index - 1] == "--method"
            for index, word in enumerate(arguments)
        ):
            reasons.append("'wget' DELETE request deletes a remote resource")
        elif executable not in allowed and ask_unknown:
            reasons.append(f"'{executable}' is not listed as safe")

    for inner in re.findall(r"\$\(([^()]*)\)", command):
        reasons.extend(_review(inner, policy, depth + 1))
    for inner in re.findall(r"(?<!\\)`([^`]*)`", command):
        reasons.extend(_review(inner, policy, depth + 1))
    return list(dict.fromkeys(reasons))


def review_reasons(command: str, policy_path=POLICY_PATH) -> list[str]:
    """Return reasons to ask for approval, using the editable policy file."""
    try:
        policy = _load_policy(policy_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"Approval policy could not be loaded: {exc}"]
    return _review(command, policy)


def approve_command(command: str, reasons: list[str], *, input_fn=input, is_tty=None) -> bool:
    """Request an explicit yes from a human at a terminal."""
    if not reasons:
        return True
    print("\n⚠️  Approval required before executing this remote command:")
    print(command)
    for reason in reasons:
        print(f"  • {reason}")
    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        print("⛔ No interactive terminal is available; command not executed.")
        return False
    try:
        answer = input_fn("Type 'yes' to execute, or anything else for a new goal: ")
    except (EOFError, KeyboardInterrupt):
        print("\n⛔ Command not approved.")
        return False
    return answer.strip().lower() == "yes"


def requests_human_confirmation(message: str) -> bool:
    """Detect a deletion-related request for human/owner approval."""
    if not message:
        return False
    text = " ".join(message.lower().split())
    deletion = r"\b(?:delet(?:e|ed|ing|ion)|remov(?:e|ed|ing|al)|eras(?:e|ed|ing|ure)|wipe|destroy|shred|unlink|rm)\b"
    actor = r"(?:system owner|owner|administrator|admin|human|user|operator)"
    approval = r"(?:confirm(?:ation)?|approv(?:e|al)|authoriz(?:e|ation)|permission|consent|review|verify|signs?[ -]?off)"
    request = r"(?:ask|have|obtain|get|request|require[sd]?|need[sd]?|must|await(?:ing)?|wait(?:ing)? for|please)"
    patterns = (
        rf"\b{request}\b.{{0,100}}\b{actor}\b.{{0,80}}\b{approval}\b",
        rf"\b{request}\b.{{0,100}}\b{approval}\b.{{0,80}}\b{actor}\b",
        rf"\b{actor}\b.{{0,80}}\b{request}\b.{{0,80}}\b{approval}\b",
        rf"\b{actor}\b.{{0,80}}\b{approval}\b.{{0,80}}\b{request}\b",
        rf"\b{approval}\b.{{0,80}}\b{actor}\b.{{0,80}}\b{request}\b",
    )
    return any(
        re.search(deletion, text[max(0, match.start() - 100):match.end() + 100])
        for pattern in patterns
        for match in re.finditer(pattern, text)
    )


def request_human_confirmation(message: str, *, input_fn=input, is_tty=None) -> bool:
    """Ask the current user to approve an LLM or command-output request."""
    print("\n⚠️  Human confirmation requested:")
    print(message)
    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        print("⛔ No interactive terminal is available; approval not given.")
        return False
    try:
        answer = input_fn("Type 'yes' if you approve, or anything else for a new goal: ")
    except (EOFError, KeyboardInterrupt):
        print("\n⛔ Approval not given.")
        return False
    return answer.strip().lower() == "yes"


def confirmation_excerpt(message: str) -> str:
    """Show the relevant request without dumping a large command output."""
    for line in message.splitlines():
        if requests_human_confirmation(line):
            return line.strip()[:500]
    compact = " ".join(message.split())
    match = re.search(r"\b(?:system owner|owner|administrator|admin|human|user|operator)\b", compact, re.I)
    start = max(0, match.start() - 120) if match else 0
    return compact[start:start + 500]

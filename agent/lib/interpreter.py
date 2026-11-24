from openai import OpenAI
import re

from lib.utils import MODELS, call_model, load_prompt


def normalize_command(command):
    """Keep nmap scans from stalling on DNS or a fragile Python formatter."""
    match = re.match(r"^\s*(nmap\b[^|]*)\s*\|\s*python(?:3(?:\.\d+)?)?\s+-c\b", command, re.DOTALL)
    scan = match.group(1) if match else command
    if match:
        scan = re.sub(r"(?<!\S)-oX\s+-(?=\s|$)", "", scan)
    if re.match(r"^\s*nmap(?:\s|$)", scan) and re.search(r"(?<!\S)-sn(?=\s|$)", scan):
        scan = re.sub(r"(?<!\S)-R(?=\s|$)", "-n", scan)
        if not re.search(r"(?<!\S)-n(?=\s|$)", scan):
            scan = re.sub(r"\bnmap\b", "nmap -n", scan, count=1)
        # A vague local-discovery request should not expand a connected /16
        # into tens of thousands of probes. Scan the local /24 of each interface.
        local_neighborhoods = (
            "$(ip -o -4 addr show scope global | "
            "awk '{split($4, p, \"/\"); split(p[1], o, \"[.]\"); "
            "print o[1] \".\" o[2] \".\" o[3] \".0/24\"}')"
        )
        scan = re.sub(
            r"\$\(\s*ip\s+-o\s+-4\s+route\s+show\s+scope\s+link\s*\|\s*awk\s+'[^']*'\s*\)",
            lambda _: local_neighborhoods,
            scan,
        )
    return scan.strip()

def get_command_from_ollama(instruction):

    # Load the prompt from the configuration file
    try:
        PROMPT = load_prompt("config/interpreterConfig.yml")
    except Exception as e:
        raise RuntimeError("interpreter.get_command_from_ollama: could not load prompt configuration.") from e

    # Call the requested model wtih the prompt
    command = call_model(
        PROMPT,
        MODELS["interpreter"],
        instruction,
        caller="interpreter.get_command_from_ollama",
    )

    # If there is an error creating the command, the exception will rise.
    # command is only returned when it has been successfully assigned
    return normalize_command(command)


def interactive_translation_session():
    """Continuously translates user instructions into commands."""
    print("\nNatural Language to Shell Command Translator (Type 'exit' to quit)")

    while True:
        instruction = input("\nEnter an instruction: ").strip()
        if instruction.lower() == "exit":
            print("Exiting translation session.")
            break

        command = get_command_from_ollama(instruction)
        if command:
            print(f"Translated Command: {command}")
        else:
            print("No valid command generated.")


# Start interactive loop
if __name__ == "__main__":
    interactive_translation_session()

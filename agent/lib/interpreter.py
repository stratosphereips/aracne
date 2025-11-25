from openai import OpenAI

from lib.utils import MODELS, call_model, load_prompt

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
    return command


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

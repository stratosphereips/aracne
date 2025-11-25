from lib.utils import MODELS, call_model, load_prompt

def summarize_context(new_context, previous_summary=None):
    """Generates an updated context summary using an LLM."""
    prompt = load_prompt("config/summarizerConfig.yml").format(
        previous_summary=previous_summary,
        new_context=new_context,
    )
    response = call_model(
        prompt,
        MODELS["summarizer"],
        new_context,
        caller="summarizer.summarize_context",
    )
    return response

# Example Usage
if __name__ == "__main__":
    previous_summary = ""
    latest_action = ''''''

    new_summary = summarize_context(previous_summary, latest_action)
    print("Updated Summary:\n", new_summary)

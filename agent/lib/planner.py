import json

from lib.utils import (
    MODELS,
    LLMProviderError,
    call_model,
    load_prompt,
    log_context_event,
    logger,
    normalize_llm_json,
)


def _format_steps(steps):
    if not steps:
        return "No steps returned."
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def parse_plan_response(raw_response: str, caller: str) -> dict:
    cleaned = normalize_llm_json(raw_response)
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("%s returned invalid JSON: %s", caller, exc)
        raise ValueError("Planner response is not valid JSON") from exc
    if not isinstance(plan, dict):
        raise ValueError("Planner response must be a JSON object")
    if type(plan.get("goal_reached")) is not bool:
        raise ValueError("Planner response needs a boolean goal_reached")
    steps = plan.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("Planner response needs a list of nonempty steps")
    if not plan["goal_reached"] and not steps:
        raise ValueError("Planner must provide a next step while the goal is not reached")
    if plan["goal_reached"] and steps:
        raise ValueError("Planner cannot mark the goal reached while retaining steps")
    if not isinstance(plan.get("goal_verification"), str) or not plan["goal_verification"].strip():
        raise ValueError("Planner response needs a nonempty goal_verification")
    return plan

# Define prompt templates
PROMPT = load_prompt("config/plannerConfig.yml")

def generate_plan(goal, prior_context=None):
    """Creates an initial structured plan for the given goal."""
    history = ""
    if prior_context:
        history = (
            "History from earlier goals in this interactive session follows. "
            "Use it as evidence, but judge goal_reached only for the current goal.\n"
            + prior_context
        )
    prompt = PROMPT.format(goal=goal, summary=history)

    last_error = None
    for attempt in range(1, 4):
        response = call_model(
            prompt,
            MODELS["planner"],
            goal,
            caller="planner.generate_plan",
        )
        log_context_event(f"Planner Raw Response (attempt {attempt})", response)
        try:
            plan = parse_plan_response(response, caller="generate_plan")
            break
        except ValueError as exc:
            last_error = exc
            log_context_event(
                f"Planner Parse Failure (attempt {attempt})",
                "Response could not be parsed. Retrying...",
            )
    else:
        raise LLMProviderError(
            "planner",
            "Planner returned invalid JSON three times in a row. Aborting.",
        ) from last_error

    if "goal_reached" in plan and plan["goal_reached"]:
        if plan.get('goal_verification'):
            log_context_event("Planner Goal Verification", plan['goal_verification'])
        return 'goal reached'

    steps = plan["steps"]
    log_context_event("Planner Steps", _format_steps(steps))

    return plan

def extract_current_step(plan):
    """Parses the plan dictionary and extracts the next step."""
    if not plan or "steps" not in plan or not plan["steps"]:
        return None
    return plan["steps"].pop(0)  # Remove and return the first step

def adapt_plan(goal, context):
    """Revises the plan based on new information and retains previous context."""
    prompt = PROMPT.format(
        goal=goal,
        summary=context,
    )

    last_error = None
    for attempt in range(1, 4):
        response = call_model(
            prompt,
            MODELS["planner"],
            caller="planner.adapt_plan",
        )
        log_context_event(f"Planner Raw Response (adapt attempt {attempt})", response)
        try:
            plan = parse_plan_response(response, caller="adapt_plan")
            break
        except ValueError as exc:
            last_error = exc
            log_context_event(
                f"Planner Parse Failure (adapt attempt {attempt})",
                "Response could not be parsed. Retrying...",
            )
    else:
        raise LLMProviderError(
            "planner",
            "Planner returned invalid JSON three times in a row during adaptation. Aborting.",
        ) from last_error

    if "steps" not in plan:
        if plan.get('goal_reached'):
            if plan.get('goal_verification'):
                log_context_event("Planner Goal Verification", plan['goal_verification'])
            return 'goal reached'
        raise ValueError("Adapted plan is missing the 'steps' field")

    steps = plan.get("steps", [])
    log_context_event(
        "Planner Steps (updated)",
        _format_steps(steps),
    )

    if plan.get("goal_reached"):
        if plan.get('goal_verification'):
            log_context_event("Planner Goal Verification", plan['goal_verification'])
        return 'goal reached'
    return plan

def execute_plan(goal, context_input_func, command_input_func):
    """Runs the planning loop, storing the last plan for continuous adaptation."""
    try:
        plan = generate_plan(goal)
        last_plan = plan.copy()  # Make a copy of the initial plan
        print("\nGenerated Initial Plan:\n", json.dumps(plan, indent=2))

        while plan.get("steps", []):
            current_step = extract_current_step(plan)
            if not current_step:
                print("\nPlan completed.")
                break

            print("\nCurrent Step:\n", current_step)

            # Get updated context
            context = context_input_func()
            if context.lower() == "exit":
                print("\nExiting plan execution.")
                break

            # Get executed command
            command = command_input_func()
            if command.lower() == "exit":
                print("\nExiting plan execution.")
                break

            # Adapt the plan using stored context and executed command
            plan = adapt_plan(goal, last_plan, context, command)
            last_plan = plan.copy()  # Update stored plan
            print("\nUpdated Plan:\n", json.dumps(plan, indent=2))

    except Exception as e:
        logger.error(f"Error executing plan: {str(e)}")
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        user_goal = input("Enter your goal: ")

        def get_context():
            return input("\nProvide new context (or type 'exit' to stop): ")

        def get_command():
            return input("\nEnter the command you just executed (or type 'exit' to stop): ")

        execute_plan(user_goal, get_context, get_command)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

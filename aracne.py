#!/usr/bin/env python3
"""
Entry point convenience wrapper to run the agent without changing directories.
"""
from pathlib import Path
import runpy
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parent
    agent_dir = project_root / "agent"
    agent_script = agent_dir / "aracne.py"

    if not agent_script.exists():
        raise FileNotFoundError(f"Could not locate agent script at {agent_script}")

    # Ensure agent modules (e.g. `lib`) remain importable when executed from the project root.
    sys.path.insert(0, str(agent_dir))
    sys.argv[0] = str(agent_script)
    runpy.run_path(str(agent_script), run_name="__main__")


if __name__ == "__main__":
    main()

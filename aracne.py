#!/usr/bin/env python3
"""
Entry point wrapper to run the agent safely across systems and Python versions.
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

    # Add both project_root and agent_dir so imports like lib and agent.lib work
    for path in (project_root, agent_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    sys.argv[0] = str(agent_script)
    runpy.run_path(str(agent_script), run_name="__main__")


if __name__ == "__main__":
    main()


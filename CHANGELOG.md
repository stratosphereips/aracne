# Changelog

## [improve_shell_interaction] — 2026-06-22

### Fixed
- **SSH session resilience:** Commands are now wrapped with `timeout … bash -c '…'` instead of being sent raw to the shell. This guarantees the remote process terminates cleanly before Aracne's own 300 s deadline, preventing the SSH session from hanging after timed-out commands that contain shell keywords (for/do/done, pipes, redirections, etc.).
- **Continue after timeout:** A timed-out command no longer aborts the entire engagement. The agent prints a warning and continues to the next planner cycle, so a single long-running scan does not prematurely end the run.

### Technical Details
- `send_ssh_command()` now wraps the user command with `timeout --signal=KILL --kill-after=10s 280s bash -c '<command>'`. The 20 s safety margin ensures the shell prompt returns before Aracne's internal timer fires. Single quotes in the command are properly escaped with `'\''`.
- The timeout stop block in `execute_agent()` replaced `break` with `continue`, so the main loop keeps iterating.

### Background
External pentesting tools often produce commands with shell keywords (for loops, case statements, pipelines). The Linux `timeout` utility can only execute binary paths, hence the `bash -c` wrapper. Without it, timed-out shell constructs leave the remote session in an unrecoverable state, causing `SSH error: Socket is closed` after the first 300 s command timeout.

## Prompt improvements — 2026-06-22

### Fixed
- **Planner JSON reliability:** Fixed YAML brace escaping in JSON examples. The original `{{{{` → `{{` escaping produced invalid `{{"steps":...}}` format that the model sometimes copied verbatim, causing `json.loads` failures. Now uses single braces `{...}` so the model sees valid JSON.
- **Planner bloat:** Consolidated `history check`, `don't repeat`, and `critical rules` sections. Removed redundant `goal_reached` instructions. Reduced from 80 lines to ~70.
- **Planner one-liner clarity:** Explicit instruction to pack work into single chains using `&&`, `;`, and pipes. Removed contradictory "one command per action" vs "do as much as possible" rule.
- **Interpreter typos:** Fixed `transtaled`, `circumtance`, `explenation`, `commmand`.
- **Final summary richness:** Replaced rigid "exactly two sentences" with `2-line executive summary + max 10 bullet findings` format. Retains the fast-scan summary but preserves accumulated intelligence from long engagements.

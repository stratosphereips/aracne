# PR: Prompt Quality Improvements

## Files Changed
- `agent/config/plannerConfig.yml` — restructured for clarity and JSON reliability
- `agent/config/interpreterConfig.yml` — typo fixes
- `agent/config/finalSummaryConfig.yml` — richer output format

---

## 1. Planner Prompt (plannerConfig.yml)

### Problem
- **Broken JSON example**: YAML `{{{{...}}}}` escaped `{{...}}` in the rendered prompt, causing the model to emit double-braced JSON that failed parsing. This was the #1 cause of "Planner returned invalid JSON" failures in long runs.
- **Bloated**: 80 lines with repeated instructions. "Don't repeat actions" appeared 3 times. LLM attention decays with prompt length.
- **Contradiction**: "One command per action only" vs "do as much as possible in a single step."

### Fix
- **Braces**: JSON examples now use single `{...}` so the model sees valid JSON.
- **Consolidated**: Merged "history check", "don't repeat", and "critical rules" into one section. Removed redundant goal_reached instructions.
- **One-liner focus**: Explicit instruction to chain operations with `&&`, `;`, pipes.
- **Typos**: Fixed "YOu" → "You".

### Before/After (key excerpt)
```diff
- - MUST BE A VALID JSON FORMAT: {{{{\"steps\": [... ]}}}}
+ - MUST BE A VALID JSON FORMAT: {"steps": [... ]}  (single line, no newlines)

- - One command per action only
- - Try to do as much as possible in a single step
+ - Each step packs as much work as possible into one bash command
+   (oneliners using pipe, &&, ; to chain operations)
```

---

## 2. Interpreter Prompt (interpreterConfig.yml)

### Problem
Multiple typos: `transtaled`, `circumtance`, `explenation`, `commmand`, `exactly wording`.

### Fix
Typos corrected. No structural changes.

---

## 3. Final Summary Prompt (finalSummaryConfig.yml)

### Problem
"Exactly two sentences of plain text" — insufficient for a penetration test report with 20+ actions of accumulated intelligence.

### Fix
Now outputs:
1. **2-line executive summary** (keeps the fast-read benefit)
2. **10-bullet compressed findings** (captures accumulated intelligence: services, hosts, data, errors, recommendations)

---

## Verification
- Planner: JSON example renders as `{"steps": [...]}` in the prompt text (single braces).
- No inline dev comments (`-- fixed`) are present in the prompt text.
- All YAML syntax is valid (indentation preserved, block scalars intact).

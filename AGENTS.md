# Shared project instructions

This file is the single source of truth for Claude Code and Codex.

## Environment and commands
- Target OS: Ubuntu 22.04. Document actual dependency versions here.
- Architecture, build, tests, coding conventions and constraints: inspect the
  project and document verified commands before implementation. Do not invent them.
- Prefer reproducible .dev/setup.sh, scripts/build.sh and scripts/test.sh where applicable.

## Continuity and preservation
- At startup read .ai/HANDOFF.md and relevant .ai/DECISIONS.md.
- Inspect git status, git diff, git diff --cached and relevant recent git log.
- Preserve all existing edits, including work by another agent. Never revert merely
  because the agent changed. Do not reset --hard, clean, stash or overwrite user work.
- Continue the recorded task and next steps; ask if intent is unclear.
- Update HANDOFF at meaningful work boundaries and before switching agents/PCs.
  Include every existing section, actual failures, test commands and results,
  branch, relevant commit, your agent name (claude/codex) and UTC timestamp.
- Record lasting architectural decisions append-only in DECISIONS.md.
- Wrapper is not an LLM: the agent must write the semantic handoff itself.
- At 85% usage prepare a checkpoint; at 90% stop starting new tasks, finish the
  current operation safely, update HANDOFF, and ask the user to exit normally.
  Thresholds are advisory, not a guaranteed automatic interrupt.
- Same PC: switching requires no automatic commit or push.
- Different PC: review explicit paths and staged diff before a WIP commit/push.
  Do not git add . or automatically commit. Check upstream divergence first.
- Never store credentials, tokens, passwords, ~/.claude or ~/.codex in Git.
- A local manager lock cannot prevent direct CLI launches or simultaneous work
  on another PC. Keep only one working agent per project.

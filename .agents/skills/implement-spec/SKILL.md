---
name: implement-spec
description: "Implement a local specification and any existing tickets in dependency order."
---

For an explicitly assigned multi-agent task, follow its supplied scope and return
protocol; do not start another agent or coordinator.

Read AGENTS.md, WORK_LOG.md, named spec and any explicitly supplied handoff/tickets. Check Git status.
One agent; dependency order; current checkout. Preserve user edits. Tickets optional.
No automatic branch, PR, worktree, commit, or delegation.
Read relevant source. Small coherent edits; smallest meaningful Docker checks.
Inspect final diff against acceptance. Report actual checks and remaining limits.
Cross-session work: update WORK_LOG.md; link the spec and any named handoff, no repetition.

## Bounded mode

Applies when requested or spec says `Mode: bounded`.

- Spec must be `ready`; check baseline/source against recorded task progress.
  Unexpected drift affecting task: stop. Preserve unrelated changes.
- Read named source first. Expand reads only to resolve local implementation.
- Edit allowed files only. Follow settled design; preserve interfaces/invariants.
  Routine coding choices and fixes within scope: proceed.
- Run exact spec checks. Never weaken tests/acceptance to obtain a pass.
  Test failure caused by this change: fix within scope, rerun affected check.
- Missing requirement, contradictory source, needed wider edit, scientific choice,
  environment failure, or repeated failed fix: stop; report evidence + next step.
  No host-test fallback, dependency install, or experiment unless authorized.
- For an assigned task, return `ready-for-review` only after required checks pass;
  otherwise report the exact blocker. Include changed files, actual commands/results
  and untested paths. Ordinary direct work has no mandatory coordinator review.
- Caveman output: short factual bullets. Think carefully; omit thought transcript,
  repeated plan, full logs. Preserve exact failure text; link large logs.

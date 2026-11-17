---
name: handoff
description: "Save a compact, persistent handoff so another session can resume the task."
---

Use `WORK_LOG.md` as the default persistent handoff. Read its current state and
check Git/source freshness before updating it. Honor an explicitly named destination;
link it from the work log rather than duplicating the same progress in both places.

Keep the handoff short: current outcome, relevant uncommitted/user changes,
validation actually run, unresolved constraints, and the next concrete action.
Include branch/revision or source/artifact links when needed to resume safely.
Replace stale current status; preserve meaningful outcome entries.

Do not copy the plan, conversation, tool calls or full logs. Link existing specs and
scientific evidence. Add coordinator/session details only for an explicitly assigned
multi-agent task, following its provided return protocol. Do not create another
workflow or declare another agent's work accepted without reviewing its evidence.

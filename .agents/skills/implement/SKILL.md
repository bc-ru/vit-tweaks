---
name: implement
description: "Implement requested work from a task, local spec, or tickets."
---

Read the request and relevant source; follow root AGENTS.md for context and validation.
The conversation is enough for a small change; do not require a spec or tickets.

Implement one coherent change at a time with one agent. Use existing Docker test
commands appropriate to the change; add a behavioral regression test for a bug
when feasible. Inspect the final diff for correctness, scope, and protocol drift.
Use a separate review only when requested or a concrete unresolved risk warrants it.

Report changes, actual validation, and remaining limits. Update WORK_LOG.md with the outcome and any next action; honor an explicitly
named handoff when resuming an existing task. Commit or open a PR only when requested.

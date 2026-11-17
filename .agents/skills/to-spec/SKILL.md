---
name: to-spec
description: "Turn the current discussion into a concise local specification."
---

Use the agreed request. Read relevant source; settle consequential uncertainty.
When specification work is requested, write `.scratch/<feature>/spec.md` or the
user's named destination. Include behavior, acceptance, scope, source pointers and
validation. Label assumptions/open questions and preserve prior decisions.
A small task needs one screen, not an interview restart or mandatory tickets.

For a bounded implementation, record the baseline, allowed edits, invariants,
exact checks and expected results. Identify the Docker image, mounts/workdir,
data/dependencies and GPU requirement where relevant. Verify commands against
source/environment; label checks not run. Never invent passing results.

For explicitly requested multi-agent work, also specify participants, authority,
finite limits and return/acceptance protocol. Do not dispatch agents on the basis
of writing a spec alone. Unresolved consequential choices must be resolved before
marking the spec ready.

Link the spec from WORK_LOG.md when needed for continuity. Do not copy planning
conversation or template instructions into the finished specification.

---
name: code-review
description: "Review a requested diff for correctness, project constraints, and agreement with the task."
---

Review with one agent by default, checking both implementation quality and task
requirements in the same pass. Delegate only if the user requests independent review.

Resolve scope first:
- Uncommitted work: `git diff HEAD` plus relevant untracked files.
- Changes since a commit/tag: `git diff <ref> HEAD`.
- Branch/PR: compare the merge-base with HEAD (`git diff <base>...HEAD`).
Use the user's explicit range when given. If unclear, inspect status and use the
obvious scope, stating it; ask only when alternatives materially change the review.
Resolve refs and check for an empty diff before deeper inspection.

Use the request as the spec; read a linked local spec when present. A missing formal
spec is a limitation, not a reason to stop. Read only applicable project rules and
source around changed behavior. Follow root AGENTS.md for navigation and evidence.

Prioritize reproducible bugs, numerical/protocol errors, missing requirements,
and meaningful regressions. Inspect tests or run a targeted check when useful.
Avoid generic smell inventories and speculative architecture improvements.

Report actionable findings by severity with file/line, trigger, consequence, and
suggested correction. Distinguish correctness bugs from maintainability suggestions.
State review scope, checks actually run, and uncovered requirements or risks.
If none were found, say so without claiming the code is proven correct.
Do not edit during a review unless fixes were requested.

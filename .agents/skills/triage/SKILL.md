---
name: triage
description: "Assess requested local tasks, verify their premise, and record the next action."
---

Read the named task or supplied queue and relevant current source. Use a local
queue only when one is requested; no remote tracker discovery is needed.

Check whether the behavior already exists and whether a reported bug has enough
information for a targeted reproduction. Use the smallest relevant check; do not
run expensive experiments to triage. Reuse existing decisions and prior findings.

Report the observed behavior, intended behavior, actionable next step or concrete
missing information. Update an existing task using its established statuses when
appropriate; apply explicit user status changes directly. Cite current paths and
symbols when they make the next action reproducible.

Do not add a rejection database, role pipeline, interview or approval round.
Ask only about a consequential missing constraint; otherwise state assumptions
and proceed. Do not implement unless asked. Save substantive findings/next action
in WORK_LOG.md, linking any explicitly maintained task instead of copying it.

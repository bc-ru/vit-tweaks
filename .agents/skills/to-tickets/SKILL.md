---
name: to-tickets
description: "Split requested substantial work into a few verifiable local tickets."
---

Use the supplied plan/spec or conversation. Split only where work can be validated
independently or needs a separate session; one ticket is acceptable. Follow real
dependencies without imposing a web-app layering model on research scripts.

Write one file per ticket at `.scratch/<feature>/issues/<NN>-<slug>.md`, preserving
existing numbering. Include the intended behavior, source/spec pointers, acceptance
criteria, validation command when known, `Blocked by:`, and `Status: ready-for-agent`
(or `needs-info` if a material requirement is unresolved).

Use repository-relative paths and symbol names to reduce rediscovery; check their
freshness on implementation. State assumptions. Do not quiz the user on routine
slice choices or require approval before this local write. Ask only if a missing
constraint changes the work. Leave any parent spec intact.

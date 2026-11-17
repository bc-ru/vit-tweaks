---
name: domain-modeling
description: "Clarify project terminology or record a consequential design decision in CONTEXT.md or an ADR."
---

Use root `CONTEXT.md` as the project glossary and `docs/adr/` for consequential
architectural decisions. Create them only when there is a real term or decision to
record; reading existing vocabulary does not require this skill.

Check ambiguous terms against relevant source and protocol. Ask only when competing
meanings change the design. Record agreed definitions briefly, keeping task progress
and implementation details in their existing docs and WORK_LOG.md.

For glossary edits, use [CONTEXT-FORMAT.md](CONTEXT-FORMAT.md). For an ADR, read
[ADR-FORMAT.md](ADR-FORMAT.md) only if the choice is hard to reverse, surprising
without context, and reflects a real tradeoff. Preserve historical decisions and
record supersession explicitly. Do not turn ordinary discussion into an interview.

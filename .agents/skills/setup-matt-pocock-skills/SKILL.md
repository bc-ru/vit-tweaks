---
name: setup-matt-pocock-skills
description: "Repair this repository local skill configuration when setup changes are requested."
---

Maintain this repository's selected local skills in `.agents/skills/`. Start with
AGENTS.md, README.md and WORK_LOG.md; inspect only configuration relevant to the request.
Qwen discovers the same tree through the relative `.qwen/skills` link. Its root
QWEN.md points to AGENTS.md. Keep one instruction source and one default work record.

Restore or adapt only the requested skills. Preserve `agents/openai.yaml` invocation
policies and supporting resources that are actually referenced. Check frontmatter,
relative links, stale project references and discovery links after changes.

Do not install a whole upstream collection, create a tracker/workflow, or change
shared user skills, MCP credentials or global settings without authorization.
Keep unrelated archived helpers inactive. Record the outcome and validation in
WORK_LOG.md; no setup transcript or extra report.

# Agent entry point

Read `README.md` and `WORK_LOG.md` first. Then read only the protocol/source needed
for the current task. Work directly in this checkout; use one agent unless the
user asks for delegation. No mandatory tickets, contracts, handoffs, or per-task reports.

## Source of truth

- Code/configs define behavior; measured artifacts define results.
- `docs/lowrank/EXPERIMENT_DESIGN_RU.md` defines the current scientific protocol;
  `docs/lowrank/AGENT_CONTRACT.md` defines its implementation requirements.
- `docs/lowrank/RUN_STAGES_RU.md` contains runnable stages 00–02.
- `docs/jax.md` describes the runtime and compatibility boundaries.
- `docs/archive/` is historical context, never current instructions.
- Report disagreements between code, measurements and docs explicitly.

## Preserve research evidence

- Seed stays 42. Never regenerate `runs/aid_split_seed42.json` or change class order
  (30 classes in the current AID artifacts). Validation selects; test is final-only.
- Never overwrite checkpoints/results or run a sweep without an explicit request.
- JAX SVD factorizes every selected matrix at every valid rank, including expanded
  storage and full rank. `all_blocks` excludes head; legacy `all` includes it.
- Preserve symmetric sqrt(S) factors, bias, stored (out,in) orientation, and NPZ schema.
- Matrix curves use analytic spectral tails; stage 02 never runs image inference.
- Keep legacy Torch and corrected JAX results separate. Historical Torch still has
  the no-saving skip; do not use it for the current low-rank protocol.
- No speculative activation-aware methods, fine-tuning or metric/protocol changes.

## Execution

Use `scripts/run.sh cpu|gpu|torch|torch-gpu COMMAND...`; it runs as the host UID/GID.
Keep the pinned container environments; do not install global host dependencies.
GPU runs require a free compute device. Check outside the sandbox when device
nodes are hidden; an in-sandbox `nvidia-smi` failure is not a driver diagnosis.
Never stop another process to free the GPU. Start with existing batch/matrix caps.

New experiment outputs: `artifacts/experiments/<id>/<stage>/`. Existing root artifact
paths are compatibility links; do not replace those links or move the frozen split.
Use a new output directory for retries. Manifest `complete` and verified inputs,
not the presence of a file, establish completion.

## Verification and work record

Before editing: check Git status, trace ownership/invariants, state the minimal
change and verification. Preserve user edits and avoid unrelated formatting.
Run `scripts/check.sh`; run the relevant container tests for behavioral changes.
A documentation-only edit needs link/diff checks, not GPU experiments.

At the end of **every substantive task**, update the tracked `WORK_LOG.md`:
- one short entry: outcome, proof/result, remaining issue;
- update current status/next action when they change;
- link code or scientific artifacts instead of copying logs or a task transcript.

Use `.scratch/` only for disposable work. Durable decisions belong next to the
code/protocol; scientific provenance belongs with results. No new agent workflow
files or skills unless they solve an explicitly requested problem.

## Tools and permissions

CodeGraph: use the existing index for code navigation; use direct reads for docs
and configs. Context7: current platform/library docs. GitHub: remote repository
operations. MCP credentials/settings are user-level configuration; never copy them
into the repo. Do not send source to external tools without task authorization.

Prevent ownership problems with the non-root launcher/devcontainers. For existing
container-owned generated files use `./fix_permissions.sh`: it repairs only local
artifact/cache/scratch trees through Docker, without host sudo or external mounts.
Do not alter system permissions, sudoers, authentication, or other repositories.
No commits/pushes/branch changes unless the user has authorized them for this work.

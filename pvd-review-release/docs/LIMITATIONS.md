# Limitations and Open Reproduction Gaps

This document states what the release does **not** establish. These limitations
are part of the artifact contract, not optional caveats.

## Current evidence boundary

The released CPU smoke test proves the central tensor-level Path-OPD semantics
and deterministic checkpoint/resume behavior on a toy model. It does not prove:

- simulator integration;
- compatibility with unpublished model weights;
- distributed training correctness at eight-rank scale;
- benchmark score reproduction;
- equivalence of reconstructed or refactored evaluators;
- reproducibility of the upstream base model or teacher training.

The three-config synthetic smoke additionally checks the declared K, horizon,
physical action prefix, optimizer, checkpoint/reload path, the released
OpenPIAdapter boundary, and one native Endpoint-DAgger loss for ManiSkill,
CALVIN, and MetaWorld. It uses only a local synthetic policy: it does not
execute an external RLinf/OpenPI checkout, external assets, a real simulator,
CUDA, or distributed code, so it does not close any of the limitations above.
The three train entrypoints expose the same proof through `--synthetic-smoke`,
which checks runner dispatch but does not change that evidence boundary.

`scripts/benchmark_environment_smoke.py` is the separate external boundary
check. It requires explicit caller paths, reports missing dependencies/assets
as `BLOCKED`, and never upgrades a reset/action probe into a training or score
claim. The release ships no independently auditable report showing all three
real simulators passed training or formal evaluation. Three local
environment-only probes were observed with caller-supplied assets: ManiSkill
reset plus one 7D action, CALVIN scene-D reset plus one 7D action, and MetaWorld
EGL reset plus one 5x4 action block. None imported an external OpenPI model,
performed an optimizer update, wrote or evaluated a checkpoint, consumed a
fixed panel, or established a GPU protocol. The probes are troubleshooting
evidence only; the release CPU environment intentionally does not vendor
simulator imports. This limited evidence does not close the simulator, GPU,
checkpoint, panel, or paper-scale limitations above.

The values under `results/reference/` are transcribed from integrity-sealed
historical reports. They are not outputs of this compact tree, and the JSON
marks that fact with `current_release_rerun: false`.

## External assets are not closed

All three benchmarks require external base models, frozen teachers,
normalization statistics, simulator/environment assets, and final checkpoints.
ManiSkill and CALVIN also require fixed evaluation panels; CALVIN requires its
task oracle and annotations. At the time of this snapshot, selected public
repository metadata is known but the complete asset closure is not:

- the ManiSkill simulator asset bundle has a pinned public dataset route and
  revision, and its repository card reports `ODC-By`. It is a directory-style
  simulator bundle: no complete directory-tree hash or per-file digest is
  recorded here, so its manifest integrity remains `unknown`;
- the CALVIN SFT model and normalization statistics have a pinned public route
  and revision, and the repository reports `Apache-2.0`. The normalization
  entry identifies `InternRobotics/InternData-Calvin_ABC/norm_stats.json`, but
  the environment, annotations, panels, and checkpoints still lack complete
  route/license/hash closure;
- the CALVIN FlowSDE teacher has a pinned route and revision, but its license
  remains unknown;
- ManiSkill and MetaWorld SFT/FlowSDE model routes now have immutable
  revisions and model-file hashes recovered from official download-cache
  metadata, but live license/API confirmation, paper checkpoints, and fixed
  panels are still unresolved. CALVIN environment assets, task annotations,
  fixed panels, paper checkpoints, and the CALVIN teacher license likewise lack
  complete route/license/hash closure.

The selected route and revision metadata was cross-checked through a
Hugging Face compatible API mirror when the canonical endpoint was unavailable.
This is not a claim that the official endpoint was downloaded successfully,
nor that a repository-level card grants rights to redistribute every contained
file. ManiSkill's card also warns that Objaverse-derived objects can carry
additional object-level terms.

The machine-readable manifests preserve known hashes while requiring the caller
to supply paths explicitly. A repository route and revision must not be read as
a blanket redistribution grant or as proof that every file is available: the
ManiSkill directory bundle still lacks a tree digest, its package-asset cache
has no public route, and most CALVIN inputs remain unresolved. They intentionally
contain no private fallback.
Consequently, paper-scale reproduction is blocked even though core reproduction
is available.

## Benchmark code closure

### ManiSkill

Portable train/evaluate runners now exist and use the released OpenPI adapter,
but they have not been run for a full GPU update or 320-row panel in this
release environment. The historical Endpoint-DAgger workers were not
byte-identical; the runner therefore preserves the common protocol rather than
claiming byte-for-byte worker identity.

### CALVIN ABC-D

The release now includes CALVIN train/evaluate entry points, environment replay,
controller-target restore/replay contracts, and per-rank environment/RNG state
in its new checkpoints. The dependency-light dry-run
does not import the simulator or CUDA stack, so it is not evidence that the real
runtime works. The historical official evaluator loaded a shared helper outside
its sealed source snapshot. A public evaluator must remove that hidden
dependency, complete an eight-rank simulator/GPU qualification, and demonstrate
parity against all saved Official-D outputs before it can be called equivalent.

### MetaWorld MT50

The historical trainer is preserved by digest, but the exact evaluator bytes
used for the paper are unavailable. A later file with the same role has a
different digest. It must not be described as the original evaluator. Recovery
of the exact source, or row-level parity of a clearly labeled reconstruction,
is required.

## Environment and packaging

- The Python package declares a compatible PyTorch range rather than a fully
  hashed dependency lock.
- The Dockerfile pins its direct Python tools and records a multi-architecture
  base-image digest, but that image has not been built or run in this release
  environment; package downloads are not wheel-hash locked.
- No container engine was available for this release validation. No Docker
  build or container run is claimed.
- The source checkout is the complete artifact. The current wheel contains the
  Python package and CLI, not the top-level configs, result JSON, or docs.
- Cross-platform behavior has not been tested; commands assume a POSIX-like
  environment.
- Exact parameter hashes from the toy smoke can differ across dependency or
  architecture versions even when all logical checks pass.

## Statistical interpretation

The formal results cover three training seeds for ManiSkill, three for CALVIN,
and two for MetaWorld. They support descriptive replication at those seeds, not
a broad estimate over training randomness. Fixed-panel paired intervals, where
reported, characterize evaluation-panel uncertainty conditional on trained
models. They are not training-seed uncertainty.

The CPU example is not a benchmark and should never be included in a performance
table.

## Scope exclusions

- LIBERO is not one of the three formal paper benchmarks and is intentionally
  absent from the result and configuration paths.
- Internal job supervisors, process monitors, accelerator ownership checks,
  experiment dashboards, repair scripts, and storage handoff logic are not
  scientific dependencies and are intentionally excluded.
- No private training log, checkpoint, dataset, credential, or machine-specific
  absolute path is included.
- No project-level open-source license is granted in this review snapshot.

## Conditions for a complete paper-scale claim

Do not change the README status to “fully reproducible” until there is evidence
for every item below:

1. Asset URLs, revisions, licenses, sizes, and hashes are complete.
2. Training and evaluation adapters for all three benchmarks run from public
   configuration only.
3. Saved final checkpoints are legally accessible without contacting the paper
   authors or exposing reviewer identity.
4. Evaluator closure and parity issues above are resolved.
5. A new machine completes clean image build, CPU smoke, one-batch benchmark
   smoke, checkpoint reload, fixed-panel subset, and report generation.
6. Full-panel outputs match the released schema and expected coverage.
7. Full paper-scale training is rerun, or any use of historical checkpoints is
   clearly distinguished from retraining.
8. The anonymous archive, Git history, hosting account, and download endpoints
   pass both automated and manual anonymity review.

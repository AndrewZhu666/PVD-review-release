# Final Reproduction Audit

This release is executable for the Path-OPD scientific core and for
dependency-light benchmark contracts. It is not a completed independent
reproduction of the paper's three benchmark results.

## Code map

| Area | Location | Function |
|---|---|---|
| Scientific objective | `src/path_opd/core.py` | Path-state supervision, endpoint exclusion, detached student path, action-expert-only loss |
| OpenPI boundary | `src/path_opd/adapters/openpi.py` | Converts RLinf/OpenPI rollouts into the released objective |
| Release identity | `src/path_opd/release_contract.py` | Asset, panel, source-tree, provenance, and checkpoint contracts |
| ManiSkill | `benchmarks/maniskill/` | K=8 train/evaluate runners and fixed-panel coverage |
| CALVIN ABC-D | `benchmarks/calvin_abc_d/` | Official-D train/evaluate runners, 8-rank protocol, resume and row coverage |
| MetaWorld MT50 | `benchmarks/metaworld_mt50/` | K=5 train/evaluate runners and reconstructed-panel provenance |
| Historical results | `results/reference/three_benchmark_results.json` | Integrity-sealed transcription; `current_release_rerun` is false |

## Verification performed

The recorded release-audit snapshot passed its source/CPU checks on 2026-09-23
with Python 3.11 and CPU PyTorch. This is historical audit evidence, not a
claim that the current shared working tree (which may contain later edits) is
green; rerun the commands below on the exact upload copy:

```text
The recorded final verification run passed the full test suite with no test
failures or dependency warnings.
ruff check --no-cache .
compileall -q src benchmarks tests scripts
audit_anonymity.py .            # no source or Git-metadata findings
path_opd.cli smoke             # PASS; exact interrupted resume
path_opd.cli benchmark smoke   # PASS; synthetic contracts only
three train --synthetic-smoke entrypoints  # PASS; runner contracts only
three evaluate --synthetic-smoke entrypoints  # PASS; evaluator contracts only
benchmark_end_to_end_smoke.py  # PASS; all six public subprocess boundaries
qualification.py               # NOT_QUALIFIED: external runtime/asset gates remain blocked
```

The public source exports are intentionally limited to the objective,
checkpoint/panel contracts, the OpenPI boundary, and the deterministic toy
proof. Configuration and asset helpers remain module-level APIs because they
are used by the CLI, tests, and benchmark launchers; no export was removed
without a call-site and contract audit.

The recorded release archive contained no tracked or archive-intended bytecode,
cache, or editable-install metadata. Local verification may recreate ignored
cache directories; inspect the exact upload copy again immediately before
upload, and do not infer archive cleanliness from the current working tree.

The release directory has a fresh, single-commit Git history attributed to an
anonymous placeholder identity and has no remote. It does not carry either
source worktree's development history. The exact upload copy and hosting
account still require the manual checks in `docs/ANONYMITY.md`.

CALVIN `train.py --help`, `evaluate.py --help`, CPU dry-run validation, and
checkpoint/resume contract tests also pass. New CALVIN checkpoints carry one
environment continuation and RNG state per distributed rank; a legacy
rank-zero-only checkpoint is rejected for the eight-rank resume path. The
deterministic CPU smoke test does not exercise a simulator or benchmark model.

The three-config synthetic smoke runs each benchmark's declared K, horizon,
physical dimensions, action prefix, optimizer, and checkpoint/reload against a
deterministic tensor policy. It also executes the released OpenPIAdapter and a
native Endpoint-DAgger SFT loss. The report explicitly distinguishes these
synthetic calls from external RLinf/OpenPI, external assets, a simulator, GPU,
or paper-scale training, all of which remain false. It is therefore
configuration-to-core evidence only.

The ManiSkill, CALVIN, and MetaWorld train entrypoints additionally dispatch
the same contract through `--synthetic-smoke`; all three return
`synthetic_runner_contract_smoke_only` with `external_runtime_executed=false`.
This verifies public runner dispatch, not real simulator execution.

The three evaluator entrypoints expose a separate
`synthetic_evaluator_contract_smoke_only` proof. It restores the synthetic
checkpoint, binds a deterministic panel identity, and checks the exact action
prefix/trace while keeping external runtime, assets, simulator, GPU, and
paper-scale flags false. This is evaluator-boundary evidence, not a benchmark
score.

The release also includes `scripts/benchmark_environment_smoke.py`, whose
report has two independent evidence channels for every benchmark. The
`algorithm_synthetic_contract` channel runs the released CPU/tensor contract;
the environment channel imports the caller-supplied RLinf/simulator and, when
qualified, performs reset plus a small action loop. Three local
environment-only probes were observed: ManiSkill reset plus one 7D action,
CALVIN scene-D reset plus one 7D action with the expected RGB/wrist/robot
observation shapes, and MetaWorld EGL reset plus one 5x4 action block. None
imported an OpenPI model, performed an optimizer update, wrote a benchmark
checkpoint, or ran the fixed panel. All real one-update/evaluation gates
therefore remain blocked. These observations keep OpenPI, training, evaluation,
GPU, and formal-benchmark flags false; they are troubleshooting context rather
than model-qualification artifacts. An overall `BLOCKED` report can therefore
coexist with an algorithm-contract `PASS`.

The shipped real train/evaluate entrypoints fail closed before importing CUDA
or simulator code unless the caller supplies the release-pinned RLinf commit,
the matching host-support patch, and the declared `rlinf-openpi` version. The
dependency-light `--dry-run` and `--synthetic-smoke` modes intentionally skip
that external gate.

## Claim boundary

The code supports claims about the released objective, declared protocol
constants, path-free checkpoint identity, explicit asset requirements,
selected pinned public asset routes, and deterministic CPU behavior. It does
not support claims that this tree has reproduced any paper benchmark table;
that GPU or eight-rank training works here; that a Docker image was built; or
that all model, simulator, panel, and RLinf assets are publicly distributable.
Selected routes were cross-checked through a compatible API
mirror when the canonical endpoint was unavailable; that metadata check is not
download or redistribution evidence.

For ManiSkill, collection-level ODC-By metadata is recorded for the public
asset collection, but it does not establish a license for every embedded
Objaverse object. The effective simulator-asset license therefore remains
unresolved until the complete bundle is reviewed file by file.

The historical numbers remain useful as reference reports only. They are not
new measurements from this release. MetaWorld remains explicitly reconstructed
and custom-panel labeled until byte identity or equivalence is established.

## Why full reproduction is currently blocked

The latest qualification invocation could not communicate with the GPU driver;
the release `.venv` uses CPU-only PyTorch (`2.11.0+cpu`,
`torch.cuda.is_available()` is false). It also has no Docker-compatible engine and no complete public
asset/license closure for the required RLinf, models, simulators, annotations,
and formal panels. Therefore the paper-scale claim is intentionally withheld
until a CUDA-enabled environment can run the declared eight-rank jobs,
checkpoint reloads, and formal evaluation panels.

Two apparently public asset routes remain specifically incomplete: the ManiSkill
simulator route is a directory bundle without a recorded tree hash or complete
per-file digest set, and the separate ManiSkill package-asset cache has no
public route. The CALVIN normalization entry now names
`InternRobotics/InternData-Calvin_ABC/norm_stats.json`; the remaining CALVIN
environment, annotation, panel, and checkpoint entries still lack complete
route/license/hash closure. These manifest entries therefore do not establish
that the complete benchmark inputs are downloadable and file-level verifiable.

## ICLR double-blind release decision

| Requirement | Evidence in this tree | Decision |
|---|---|---|
| Reviewer can inspect the scientific core | `src/path_opd/core.py`, invariant tests, and CPU smoke contract | Supported |
| Reviewer can run a deterministic local proof | `path_opd smoke` and `tests/test_toy_smoke.py` | Supported; CPU-only |
| Reviewer can identify the declared benchmark protocol | `configs/*.json` and `configs/assets/*.json` | Supported; caller-supplied assets |
| Reviewer can independently run paper-scale benchmarks | Runner source exists and selected one-step environment probes are documented, but simulator/model/panel closure and GPU qualification are absent | Not supported yet |
| Anonymous archive is free of local build state | `.gitignore`, Docker deny-by-default context, and generated-file audit | Must be checked at archive time |
| Historical numbers are new measurements | `results/reference/three_benchmark_results.json` explicitly sets `current_release_rerun=false` | Must not be claimed |

The release is therefore suitable as an anonymous review artifact for checking
the method and its contracts, but it is not evidence of a complete independent
reproduction of the paper's benchmark tables. A submission should retain this
boundary in the README and supplementary material until the blocked runtime,
asset, evaluator, and multi-GPU gates have independent artifacts.

## Delivery evidence

The recorded independent release workspace was checked without modifying either
source worktree. That historical check confirmed the CPU smoke report, three
benchmark contract descriptions, an empty anonymous scan, and an archive tree
without bytecode, cache, or editable-install metadata. A wheel built from that
snapshot was installed into a fresh Python 3.12 environment and its CLI
produced `status=PASS` with `interrupted.exact_resume=true`. A fresh
single-commit anonymous Git history was present and had no remote. Re-run these
checks on the current exact upload copy; the source still requires an external
container-capable machine for Docker build evidence.

## Archive hygiene status

Verification commands can regenerate local build state such as `.ruff_cache`,
`__pycache__`, `*.egg-info`, and bytecode files. These paths are ignored by the
source and Docker context rules. The exact anonymous archive copy was built
from the release allowlist and checked without those paths; repeat its scan
immediately before uploading it, and recheck the hosting account and generated
archive metadata separately.

The qualification environment had no Docker-compatible engine. The release
Python environment was CPU-only, and the latest `nvidia-smi -L` invocation
failed to communicate with the driver. Consequently, Docker build/run and GPU
benchmark claims remain intentionally unqualified in this final report.

## Reviewer handoff

For an ICLR double-blind submission, describe this artifact as an anonymous
review release containing the method implementation, deterministic CPU proof,
benchmark contracts, and qualification procedures. Do not describe the
historical result table as a rerun. A reviewer can begin with the Quick start
section in `README.md`, then inspect `docs/REPRODUCIBILITY.md` and run the
contract-specific dry runs. Full benchmark execution remains conditional on
caller-supplied assets and a compatible container/GPU environment.

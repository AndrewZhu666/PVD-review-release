# Qualification Audit

This document records what the release can and cannot support as of
2026-09-23. It is deliberately narrower than a paper-results claim: a
passing source or CPU check is not evidence that a simulator, CUDA runtime,
model asset, or full benchmark run works.

Regenerate the machine-readable matrix from the repository root:

```bash
python scripts/qualification.py \
  --format json \
  --output artifacts/qualification.json
```

Use `--strict` in CI when every external gate is expected to be closed. Until
then the default command returns zero after reporting `NOT_QUALIFIED`, which
lets reviewers inspect a partially provisioned machine without hiding missing
dependencies. `--skip-runtime` performs only static inspection.

## Current evidence

The recorded qualification snapshot is `NOT_QUALIFIED`: its source/CPU checks
passed while external container/GPU/runtime/asset gates remained blocked. This
status applies to the audited source snapshot; rerun the matrix on the exact
release archive before making a qualification claim. Running verification may
recreate ignored caches and artifacts, so inspect the generated archive
separately.

| Qualification level | Current result | Evidence |
|---|---|---|
| Source layout and contracts | Pass | Required source, config, Docker, audit, and test files are present. |
| Deterministic CPU core | Pass | Toy optimization, frozen-teacher checks, checkpoint save/reload, and exact interrupted resume complete. |
| Three-config synthetic contract smoke | Pass | All three formal configurations execute the core objective, released OpenPIAdapter boundary, native Endpoint-DAgger loss, action crop, optimizer, and checkpoint/reload on synthetic tensors; external OpenPI, simulator, assets, GPU, and paper-scale execution are explicitly false. |
| Three-runner synthetic entrypoint smoke | Pass | ManiSkill, CALVIN, and MetaWorld train entrypoints each dispatch the synthetic contract report; `external_runtime_executed` remains false. |
| Three-evaluator synthetic boundary smoke | Pass | ManiSkill, CALVIN, and MetaWorld evaluator entrypoints run the synthetic train/checkpoint proof, validate a deterministic panel denominator/digest, and verify exact action traces; all external-runtime flags remain false. |
| Six-entrypoint aggregate synthetic smoke | Pass | One command launches all three train and all three evaluate entrypoints as subprocesses, then validates exact resume, complete panel consumption, checkpoint schema, ordered identity binding, and action traces. |
| Environment probe fail-closed contract | Pass | `scripts/benchmark_environment_smoke.py` runs the synthetic algorithm contract before preflight and reports missing external paths as `BLOCKED` without guessing host locations. |
| Unit and contract tests | Recorded pass | The audit snapshot's isolated suite had no test failures or dependency warnings; rerun after any working-tree edits. |
| Static style and compilation | Recorded pass | Ruff and `compileall` passed for the audited snapshot; rerun on the exact upload copy. |
| Anonymous source and Git scan | Pass | Both the source-only scan and `scripts/audit_anonymity.py . --format json` returned no findings. The repository has one anonymous root commit and no remote. |
| Docker context allowlist | Pass (static) | The context is deny-by-default and explicitly excludes local virtual environments and caches. |
| Docker base reference | Pass (static) | The multi-architecture base image reference contains a complete OCI index digest; no build is claimed. |
| Clean container build/run | Blocked | No clean container build/run is recorded for this release. |
| GPU qualification | Blocked | No CUDA training qualification is recorded; CPU-only checks do not establish GPU support. |
| ManiSkill real one-update/evaluation | Blocked | No qualifying run combines the external OpenPI model, a real optimizer update, checkpoint save/reload, and evaluation. The declared GPU task also requires the complete `bridge_v2_real2sim` and package-asset bundles. |
| MetaWorld real one-update/evaluation | Blocked | Real GPU training, checkpoint save/reload, and custom-panel evaluation have not been demonstrated together for this release. |
| CALVIN source and dry-run contract | Pass | Train/evaluate entry points, explicit asset paths, restore/replay state, and dependency-light dry-run checks are present. |
| CALVIN real one-update/Official-D evaluation | Blocked | No qualifying run combines a real OpenPI update, checkpoint save/reload, and Official-D evaluation; GPU evaluator parity is not established. |
| Paper-scale result reproduction | Blocked | No eight-rank/full-budget/full-panel run has been completed from this release tree. |

## Dependency and asset closure

The Docker image is intentionally CPU-only and covers the scientific core. It
does not install any benchmark simulator. The benchmark qualification image or
lock file still needs exact, tested versions for RLinf/OpenPI, PyTorch/CUDA,
NumPy, Gymnasium, ManiSkill, MetaWorld, MuJoCo, and CALVIN.

Every required entry in `configs/assets/*.json` is currently caller supplied.
At audit time the manifests recorded:

| Benchmark | Required assets | Known digest | Public route entries | Known revision | Known license |
|---|---:|---:|---:|---:|---:|
| ManiSkill | 12 | 10 | 4 | 4 | 0 |
| CALVIN ABC-D | 12 | 10 | 3 | 3 | 2 |
| MetaWorld MT50 | 7 | 7 | 3 | 3 | 0 |

A digest alone is insufficient for independent reproduction. Each required
asset must also have a reviewer-safe acquisition route, revision, license, and
successful preflight verification. The listed routes and revisions are
caller-supplied metadata; they do not download files or grant blanket
redistribution rights. Selected ManiSkill/CALVIN repository routes and
revisions were cross-checked through a Hugging Face-compatible API mirror when
the canonical endpoint was unavailable; direct download is not claimed here.
The ManiSkill simulator route is a directory bundle with no complete tree hash
or per-file digest set, and its separate package-asset source has no public
route. The CALVIN normalization entry names
`InternRobotics/InternData-Calvin_ABC/norm_stats.json`; live license/API
confirmation and complete model-directory closure remain pending. The
environment, annotation, panel, and checkpoint entries are also not closed.
These entries do not establish file-level asset closure. The complete
MetaWorld/CALVIN environments, fixed panels, paper checkpoints, and a
Git-verifiable pinned RLinf/OpenPI checkout still require public, verifiable
acquisition details.

## Executable qualification matrix

Run these commands in order on a clean review machine. A broad claim may only
use evidence from its own row or a stronger row.

| Gate | Command | Required result | Supports |
|---|---|---|---|
| Static audit | `python scripts/qualification.py --skip-runtime --strict` | No WARN, BLOCKED, or FAIL after provisioning | Source packaging only |
| CPU test suite | `python -m pytest -p no:cacheprovider -q` | All tests pass | Core and contract verification |
| Style | `python -m ruff check --no-cache .` | Clean | Source quality only |
| Compilation | `python -m compileall -q src benchmarks tests` | Exit 0 | Import/syntax coverage only |
| CPU reproduction | `python -m path_opd.cli smoke --work-dir artifacts/toy-smoke --output artifacts/toy-smoke/report.json` | `status=PASS` and `interrupted.exact_resume=true` | Deterministic core reproduction |
| Three-config synthetic smoke | `python -m path_opd.cli benchmark smoke --benchmark all --work-dir artifacts/benchmark-smoke --output artifacts/benchmark-smoke/report.json` | `status=PASS`, `synthetic_only=true`, synthetic adapter/baseline flags true, and all external-runtime flags false | Config-to-core contract only; not benchmark qualification |
| Runner synthetic smoke | `python benchmarks/<benchmark>/train.py --synthetic-smoke` for each benchmark | `status=PASS`, `claim=synthetic_runner_contract_smoke_only`, `external_runtime_executed=false` | Public runner dispatch only; not benchmark qualification |
| Evaluator synthetic smoke | `python benchmarks/<benchmark>/evaluate.py --synthetic-smoke` for each benchmark | `status=PASS`, `claim=synthetic_evaluator_contract_smoke_only`, checkpoint/panel/action-trace checks true, and external-runtime flags false | Synthetic train/checkpoint/evaluator boundary only; not benchmark qualification |
| Aggregate six-entrypoint smoke | `python scripts/benchmark_end_to_end_smoke.py --work-dir artifacts/three-benchmark-e2e-smoke --output artifacts/three-benchmark-e2e-smoke/report.json` | `status=PASS`, all three train/evaluate summaries pass, and every external-runtime flag is false | Public subprocess wiring only; not simulator or paper-result evidence |
| Environment probe contract | `python scripts/benchmark_environment_smoke.py --benchmark all --output artifacts/environment-smoke/report.json` | Algorithm contract `PASS` for every benchmark and missing external paths reported as `BLOCKED` | Fail-closed preflight behavior; not simulator or paper-result evidence |
| Anonymous source audit | `python scripts/audit_anonymity.py . --no-git --format json` | Empty JSON array | Source-tree hygiene only |
| Docker build | `docker build --pull --no-cache -f docker/Dockerfile -t path-opd-review:qualification .` | Successful clean build; record image digest | CPU container availability |
| Docker CPU run | `docker run --rm path-opd-review:qualification smoke` | PASS report | Isolated CPU core reproduction |
| Docker tests | `docker run --rm path-opd-review:qualification test` | All tests pass | Isolated source/contract verification |
| Docker audit | `docker run --rm path-opd-review:qualification audit` | No findings | Files copied into image are clean |
| GPU visibility | `nvidia-smi -L` and `python -c "import torch; assert torch.cuda.is_available()"` | Both pass | Prerequisite only |
| Benchmark assets | `path-opd benchmark describe --benchmark <name>` followed by `path-opd benchmark preflight --benchmark <name> --purpose <train|evaluate> --asset NAME=PATH ...` | All known hashes verified; no undeclared host paths | Asset identity for one operation |
| ManiSkill qualification | Eight-rank train with one update, save/reload, then a clearly labeled subset evaluation | Successful logs, checkpoint reload, row coverage, peak memory | ManiSkill runner qualification only |
| MetaWorld qualification | `MUJOCO_GL=egl` eight-rank one-update train, save/reload, subset evaluation | Successful EGL render, logs, reload, row coverage, peak memory | Reconstructed MetaWorld runner qualification only |
| CALVIN qualification | Eight-rank one-update train, save/reload controller target, Official-D subset evaluation | Successful restore/replay, evaluator rows, peak memory | CALVIN runner qualification only |
| Paper reproduction | Full declared seeds, 80k chunks, 10k updates, and formal evaluation panels | Complete artifacts and aggregate results under pinned stack | Paper-result reproduction |

The one-update rows must record exact command lines, code revision, framework
revision, asset hashes, accelerator and driver versions, peak device memory,
wall time, checkpoint hash, resume status, evaluation coverage, and failure
logs. A successful one-update qualification is not a full-paper rerun.

## Release hygiene before archival

Verification commands can recreate local generated state such as `.ruff_cache`,
`__pycache__` directories, `src/path_opd.egg-info`, and bytecode files. These
paths are excluded from the Docker context, but must not be included in the
anonymous archive. The anonymous Git identity does not establish archive
hygiene. Repeat the check on the exact archive copy immediately before submission. Also
verify manually that the archive has no Git metadata,
absolute symlinks, credentials, experiment-tracker files, PID/cgroup data,
private checkpoints, or machine-specific paths.

The Docker base image is pinned to a recorded multi-architecture digest, but
the image has not been built for this release claim. Python package downloads are
version-pinned but not wheel-hash-pinned. Before calling the container
immutable, build and run it on a container-capable machine and use a lock or
constraints file with hashes. The project-level license decision also remains
outside this qualification audit.

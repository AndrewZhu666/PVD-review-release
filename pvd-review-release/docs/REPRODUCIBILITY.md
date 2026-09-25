# Reproducibility Guide

This guide separates four different claims that are easy to conflate:

1. **Core reproduction** executes the Path-OPD objective on CPU.
2. **Source-contract verification** checks fixed vectors, gradients, configs,
   asset manifests, and checkpoint/resume behavior.
3. **Historical result verification** checks the released transcription and
   provenance of the paper metrics; it does not rerun a simulator.
4. **Paper-scale reproduction** retrains and evaluates the three benchmarks.
   This fourth level is not currently closed by the public tree.

The first two levels are runnable without model or simulator assets. The third
is inspectable. The fourth requires the missing items listed below and in
[Limitations](LIMITATIONS.md).

## 1. Clean Python environment

Prerequisites:

- Python 3.10 or newer;
- a POSIX-like shell for the commands below;
- internet access to install PyTorch, unless the dependency is already cached;
- no GPU for the core reproduction.

From a fresh checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

The test suite is designed to cover:

- fixed-vector action cropping and weighted loss parity;
- the archived K=4 and K=8 flow-time grids;
- endpoint exclusion and detached path states;
- teacher evaluation/no-gradient/immutability guards;
- action-expert-only parameter selection;
- OpenPI adapter shape and context behavior;
- benchmark configuration and asset-manifest schema validation;
- historical result transcription and anonymity checks.

The exact number of tests can grow as release closure improves, so a hard-coded
test count is not part of the scientific contract.

### Keep the benchmark runtime separate

The environment above is the portable core environment. It follows
`uv.lock` and uses CPU PyTorch `2.11.0`. Real benchmark workers use the
caller-owned RLinf/OpenPI environment instead: the pinned PyPI artifact
`rlinf-openpi==0.1.1` declares `torch==2.7.1` and additional CUDA/JAX and
simulator dependencies. Install that artifact only in the RLinf environment;
do not add it to the core `.venv`, where it would conflict with the lock file.
From the benchmark environment, make this source checkout importable without
re-solving its core dependencies:

```bash
export PYTHONPATH=/path/to/path-opd-review/src:/path/to/path-opd-review
python -m pip install "rlinf-openpi==0.1.1"
```

The baseline checkout, patch, simulator versions, and model/asset paths remain
caller supplied as described in the root README's RLinf setup and asset sections.
The synthetic and CPU commands in this guide continue to use only the core
environment.

## 2. Deterministic CPU proof

Run:

```bash
python -m path_opd.cli smoke \
  --work-dir artifacts/toy-smoke \
  --output artifacts/toy-smoke/report.json
python -m json.tool artifacts/toy-smoke/report.json
```

The smoke runner creates the same student twice. One copy trains without an
interruption; the other saves optimizer, random-generator, model, configuration,
and teacher-integrity state halfway through, reloads, and finishes. It fails if
the resumed student parameter hash differs from the uninterrupted run.

Required report predicates:

```text
status                                      PASS
checks.loss_decreased                       true
checks.teacher_unchanged                    true
checks.endpoint_excluded                    true
checks.path_states_detached                 true
interrupted.exact_resume                    true
```

The report also includes configuration, initial/final loss, and parameter
hashes. Treat exact floating-point values and hashes as stack-specific unless
the same Python/PyTorch/container versions are used. The equality between
uninterrupted and resumed execution is asserted within the same run.

For an alternate deterministic seed or a shorter diagnostic:

```bash
python -m path_opd.cli smoke --seed 11 --updates 8 --work-dir artifacts/smoke-11
```

### Three-config synthetic contract smoke

Run the released core with the exact flow and action dimensions declared by all
three benchmark configurations:

```bash
python -m path_opd.cli benchmark smoke \
  --benchmark all \
  --work-dir artifacts/benchmark-smoke \
  --output artifacts/benchmark-smoke/report.json
```

This command executes Path-OPD optimization, the repository's OpenPI adapter,
one native Endpoint-DAgger SFT loss, physical-prefix consumption, and
checkpoint/reload on deterministic synthetic tensors. A valid report says
`claim: "synthetic_contract_smoke_only"`, `synthetic_only: true`, and
`checkpoint_reload_executed: true`. It also says
`synthetic_openpi_adapter_executed: true` and
`synthetic_endpoint_dagger_executed: true`, while
`external_openpi_executed: false` and
`external_endpoint_dagger_executed: false`. External assets, real simulators,
GPU, and paper-scale execution remain false. `exact_resume` is true only when
the student, optimizer, and smoke RNG states all match the uninterrupted run.
Consequently, a PASS here catches configuration and core integration drift but
does not qualify any real benchmark runner or reproduce any paper metric.

For a runner-boundary check, execute the corresponding train entrypoint:

```bash
python benchmarks/maniskill/train.py --synthetic-smoke
python benchmarks/calvin_abc_d/train.py --synthetic-smoke
python benchmarks/metaworld_mt50/train.py --synthetic-smoke
```

Each report has `claim: "synthetic_runner_contract_smoke_only"` and
`external_runtime_executed: false`. This verifies that the public runner
entrypoint reaches the released objective, adapter, baseline, checkpoint, and
action contract. It remains a synthetic test: real simulator, external
RLinf/OpenPI, GPU, and paper-scale evidence require the assets and environment
listed below.

For a single subprocess-level check across all public train/evaluate commands,
run:

```bash
python scripts/benchmark_end_to_end_smoke.py \
  --work-dir artifacts/three-benchmark-e2e-smoke \
  --output artifacts/three-benchmark-e2e-smoke/report.json \
  --updates 2 \
  --batch-size 1 \
  --seed 7
```

The command executes all three train entrypoints and all three evaluator
entrypoints, then verifies optimizer updates, exact model/optimizer/RNG resume,
checkpoint schema, every synthetic panel row, ordered panel identity binding,
and exact action traces. The report is intentionally labelled
`synthetic_three_benchmark_end_to_end_only`; a pass is not simulator or
paper-metric evidence.

### One-step environment qualification probe

The environment probe is a separate, explicitly parameterized command. Supply
all external locations rather than relying on machine-local defaults:

```bash
python scripts/benchmark_environment_smoke.py \
  --benchmark all \
  --rlinf-checkout /path/to/RLinf \
  --maniskill-simulator-assets /path/to/maniskill-task-assets \
  --maniskill-package-assets /path/to/maniskill-package-cache \
  --calvin-environment-assets /path/to/calvin-scene-and-data \
  --metaworld-task-config /path/to/RLinf/rlinf/envs/metaworld/metaworld_config.json \
  --steps 1 \
  --output artifacts/environment-smoke/report.json
```

The script first runs the synthetic algorithm contract for each selected
benchmark in a temporary directory. It then performs reset/action probing only
when the selected RLinf checkout, dependencies, and caller-supplied assets pass
preflight. It never downloads, copies, or recursively hashes those assets.
The machine-readable schema is
`path-opd-benchmark-environment-smoke-v1`.

The report records an observed RLinf Git revision when the supplied directory
is a Git checkout, but this probe does not apply or verify the host patch. Use
the pinned revision and `git apply --check` procedure in the root README as a
separate runtime gate. Add `--require-pinned-rlinf` when the smoke itself must fail
closed unless the checkout's Git HEAD matches the recorded revision; the
default is a presence/compatibility probe for copied source directories.

Interpret the fields independently:

```text
results[*].checks.algorithm_synthetic_contract.status = PASS
results[*].evidence.environment_probe_executed = true|false
results[*].evidence.external_runtime_executed = true|false
results[*].evidence.external_assets_used = true|false
results[*].evidence.real_simulator = true|false
results[*].evidence.gpu_executed = true|false
results[*].evidence.one_step_smoke_only = true  # when --steps 1
results[*].evidence.short_rollout_smoke_only = false  # when --steps 1
results[*].evidence.one_step_smoke_only = false  # when --steps >1
results[*].evidence.short_rollout_smoke_only = true  # when --steps >1
results[*].evidence.formal_benchmark = false
training_or_evaluation_executed = false
paper_scale = false
```

Thus an overall `BLOCKED` report with algorithm-contract `PASS` means the
released code path was checked but an external simulator gate was unavailable;
it is not an algorithm failure and it is not a paper result. Running without
paths intentionally exercises this fail-closed behavior. `--strict` returns a
non-zero status for `BLOCKED` or `FAIL`; `--skip-algorithm-smoke` is an explicit
environment-only mode and removes the algorithm evidence.

The default `--steps 1` command is the one-step smoke described above. A small
larger value is allowed for diagnostics and is labelled
`synthetic_algorithm_plus_environment_short_rollout_smoke`; it remains a
short probe, never a training or evaluation run.

No independently auditable report proving three real simulator training or
evaluation runs is included in this release. Current qualification therefore
leaves every real one-update/evaluation gate blocked. Three local
environment-only probes were observed with caller-supplied assets: ManiSkill
reset plus one 7D action, CALVIN scene-D reset plus one 7D action, and MetaWorld
EGL reset plus one 5x4 action block. None executed an OpenPI model, optimizer
update, checkpoint write/reload, fixed panel, GPU training protocol, or
benchmark metric. Re-run the command above in the exact external environment
and preserve its JSON output before making a simulator-wiring claim. The core
CPU environment intentionally does not vendor those simulator imports, so the
same command may still be `BLOCKED` there before the external runtime is
imported.

The JSON report records caller-supplied paths to make diagnostics reproducible;
it declares `report_may_contain_caller_paths: true`. Redact those path fields
before publishing a runtime report from a private machine. The release source
itself contains no machine-specific fallback paths.

## 3. Docker environment

Build and run from the repository root:

```bash
docker build --pull -f docker/Dockerfile -t path-opd-review:local .
docker run --rm path-opd-review:local
docker run --rm path-opd-review:local test
docker run --rm path-opd-review:local audit
```

The build context is allowlisted by `docker/Dockerfile.dockerignore`. This
prevents caches, local artifacts, Git metadata, and unlisted files from being
sent to the builder. The runtime user is unprivileged. The default `smoke`
command writes temporary checkpoint state inside the container and emits the
report on standard output.

The `python:3.11-slim-bookworm` base is pinned to OCI index digest
`sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b`.
The digest was cross-checked on 2026-09-22 against the Docker Official Images
`repo-info` snapshot at commit `40cdaf34215d6a292f2c62798b194d67d4b1ad55`.
This records the selected multi-architecture input; it is not evidence that the
image was built. Build and run it on a clean container-capable machine before
claiming container validation.

To retain artifacts, create a writable host directory and run with the host
user identity:

```bash
mkdir -p artifacts/docker-smoke
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD/artifacts/docker-smoke:/workspace/artifacts" \
  path-opd-review:local smoke \
  --output /workspace/artifacts/report.json
```

### Docker validation status

The Dockerfile and entrypoint receive static checks, and their underlying
Python commands execute in an isolated virtual environment. No Docker,
Podman, Buildah, or Apptainer execution is claimed for this release. A
maintainer must verify
the pinned base through a registry-capable client and run all three commands
above on a clean machine before marking container validation complete. Record
the resulting image ID and full command output in the release report.

## 4. Algorithm-to-code map

| Scientific operation | Released implementation | Verification |
|---|---|---|
| define executable action domain | `ActionContract` | fixed-vector crop test |
| define pre-transition times/weights | `FlowSchedule` | archived time-grid tests |
| exclude endpoint and detach path | `RolloutTrace.from_chains` | gradient and sentinel tests |
| freeze and guard teacher | `FrozenTeacher` | mode, gradient, mutation tests |
| query identical student states/times | `PathOPD.supervise` | query identity tests |
| time-weighted valid-domain MSE | `PathOPD.loss_from_velocities` | literal loss/gradient oracle |
| train action expert only | `configure_action_expert` | selector tests |
| bind RLinf/OpenPI calls | `OpenPIAdapter` | adapter contract tests |

The CPU toy model is deliberately small, but it invokes the same public core
that the OpenPI adapter invokes. There is no separate demonstration-only loss.

## 5. Historical reference results

Validate JSON syntax and inspect the evidence label:

```bash
python -m json.tool results/reference/three_benchmark_results.json >/dev/null
python - <<'PY'
import json
from pathlib import Path

data = json.loads(
    Path("results/reference/three_benchmark_results.json").read_text(encoding="utf-8")
)
print(data["artifact_type"])
print(data["evidence"])
PY
```

The expected artifact type is `historical_sealed_reference_results`, and
`evidence.current_release_rerun` is `false`. This flag must not be changed until
the corresponding release commands have actually reproduced every reported
row under the recorded protocols.

## 6. Benchmark protocols

Authoritative release-side contracts are:

| Benchmark | Config | Asset manifest |
|---|---|---|
| ManiSkill | `configs/maniskill.json` | `configs/assets/maniskill.json` |
| CALVIN ABC-D | `configs/calvin_abc_d.json` | `configs/assets/calvin_abc_d.json` |
| MetaWorld MT50 | `configs/metaworld_mt50.json` | `configs/assets/metaworld_mt50.json` |

Common training contract:

- methods: Endpoint-DAgger and Path-OPD;
- 8 ranks, global batch 8;
- 80,000 fresh online chunks and 10,000 optimizer updates;
- no replay and one presentation per chunk;
- AdamW with learning rate `7.91e-6`, betas `(0.9, 0.95)`, epsilon `1e-5`,
  weight decay `0.01`, and global gradient clipping at `1.0`;
- checkpoints after 10k, 20k, 40k, 60k, and 80k chunks;
- only the action expert is trainable; the teacher is frozen and actor-only.

Benchmark-specific contract:

| Benchmark | Training seeds | K | Horizon/prefix/dim | Evaluation |
|---|---|---:|---|---|
| ManiSkill | 0, 1, 2 | 8 | 8 / 5 / 7 | 320 fixed resets, 80 primitive steps |
| CALVIN ABC-D | 0, 1, 2 | 8 | 5 / 5 / 7 | 1,000 Official-D sequences; training episode limit 480 primitive steps, evaluation subtask limit 360 primitive steps |
| MetaWorld MT50 | 0, 1 | 5 | 5 / 5 / 4 | 50 tasks x 10 variants, 160 steps |

The manifests list base models, frozen teachers, normalization statistics,
environment assets, panels, and paper checkpoints. Public routes are now
identified for the ManiSkill and MetaWorld SFT/FlowSDE model names as well as
the previously recorded ManiSkill asset bundle and CALVIN model/statistics
entries. Official download-cache metadata records immutable revisions and
model-file hashes for the ManiSkill and MetaWorld models, while live Hub/API
and license confirmation remain pending. The first ManiSkill repository card
reports `ODC-By`; the CALVIN SFT repository reports `Apache-2.0`; several
model licenses remain unknown. A known SHA-256 is retained where evidence
supports one. These entries are caller-supplied metadata, not automatic
downloads or blanket redistribution grants. A known hash proves expected bytes
only after the file is obtained and does not provide a license grant. The
ManiSkill route names a directory-style simulator bundle, but this release has
neither a complete directory-tree hash nor per-file digests for it, so its
manifest integrity remains `unknown`; the separate package-asset cache has no
public route. The CALVIN normalization entry identifies the concrete nested
file `InternRobotics/InternData-Calvin_ABC/norm_stats.json`; the ManiSkill and
MetaWorld model normalizer paths are recorded in their manifests, while the
remaining CALVIN environment, annotation, panel, and checkpoint entries stay
unresolved. None of these entries claims that the complete benchmark input set
can be downloaded and verified from this tree.

MetaWorld has one additional framework-owned input: the ordered 50-task prompt
mapping `metaworld_config.json`. It is intentionally resolved from the pinned
RLinf checkout rather than copied into this repository. The runner accepts the
`rlinf/envs/metaworld/metaworld_config.json` layout and the supported
`rlinf/envs/sim/metaworld/metaworld_config.json` fallback, hashes the selected
file into every checkpoint and panel identity, and stops if neither file exists.
This is part of the RLinf source revision listed above, not a private host
default or an untracked local asset.

For the framework/package pins and the evidence behind each public route, see
[`External asset and framework sources`](EXTERNAL_ASSET_SOURCES.md). That note
also distinguishes the `rlinf-openpi` PyPI artifact from a Git revision and
records the license caveats for ManiSkill's package and object assets.

### Asset preflight

Before invoking a benchmark runner, inspect the validated contract to obtain the
exact manifest parameter names:

```bash
path-opd benchmark describe --benchmark maniskill
```

Then provide every required path explicitly. The assignment names are the
`parameter` values printed by `describe`; repeat `--asset` once per asset:

```bash
path-opd benchmark preflight \
  --benchmark maniskill \
  --purpose train \
  --asset maniskill_base_model=/review-assets/maniskill/base-model \
  --asset maniskill_frozen_teacher=/review-assets/maniskill/frozen-teacher \
  --asset maniskill_normalization_stats=/review-assets/maniskill/norm.json \
  --asset maniskill_simulator_assets=/review-assets/maniskill/simulator-assets \
  --asset maniskill_package_assets=/review-assets/maniskill/package-assets \
  --output artifacts/maniskill-preflight.json
```

The example paths are placeholders and are not supplied by this repository.
`preflight` verifies known SHA-256 values and reports missing public URL,
revision, or license metadata; it does not download assets or establish that a
simulator, GPU, RLinf checkout, or benchmark evaluator can run. A result with
`status: "PASS_WITH_UNVERIFIED_ASSETS"` means that supplied files existed but
at least one required manifest entry has no recorded digest, so it is not
evidence of paper-scale reproduction. For `reproduce_paper_results`, add
`--method path_opd --seed 0` (or the declared method/seed pair) so the
corresponding sealed checkpoint is selected; do not use this purpose unless
the checkpoint and panel are legally available to the reviewer.

### Real runner command templates

After a successful asset preflight, the following commands are the explicit
entry points for a one-update qualification run. Every /path/to/... value is
caller supplied; the release has no host-specific fallback. Run the matching
--dry-run command first when checking argument names and output layout. A
one-update run is a runtime qualification artifact, not a paper result. The
MetaWorld dry-run templates use repository-local `artifacts/` outputs so they
have a writable existing ancestor when run from a writable checkout; dry-run
does not create those output directories.

The runners place each checkpoint under
`<output>/checkpoints/chunks_<global_chunks>/`. The templates use
`chunks_000008` for the one-update qualification run; use `chunks_080000` for
the full 80,000-fresh-chunk budget. ManiSkill stores `checkpoint.pt`, while
CALVIN stores `trainer_state.pt` and MetaWorld stores a checkpoint directory
containing `model.safetensors` and `trainer_state.pt`.

ManiSkill training (eight ranks):

~~~bash
torchrun --standalone --nproc_per_node=8 \
  benchmarks/maniskill/train.py \
  --method path_opd --seed 0 --max-updates 1 \
  --rlinf-root /path/to/RLinf \
  --base-model /path/to/maniskill/base-model \
  --teacher-model /path/to/maniskill/teacher-model \
  --normalization-stats /path/to/maniskill/norm.json \
  --teacher-normalization-stats /path/to/maniskill/teacher/actor/assets/global_step_150/meta/norm_stats.json \
  --simulator-assets /path/to/maniskill/task-assets \
  --maniskill-package-assets /path/to/maniskill/package-assets \
  --output-dir /path/to/runs/maniskill/path-opd-seed0
~~~

ManiSkill has no published formal panel digest in this release. For a small
real-runtime qualification, use a caller-created 320-row panel and label it
`custom`; custom does not mean fewer rows, but it does mean this is not a
paper-panel result. The caller must create the panel using the exact schema
accepted by `benchmarks/maniskill/common.py` (unique reset IDs, environment ID,
and ownership of 40 rows per rank), record its SHA-256, and supply it below:

~~~bash
torchrun --standalone --nproc_per_node=8 \
  benchmarks/maniskill/evaluate.py \
  --rlinf-root /path/to/RLinf \
  --base-model /path/to/maniskill/base-model \
  --normalization-stats /path/to/maniskill/norm.json \
  --teacher-normalization-stats /path/to/maniskill/teacher/actor/assets/global_step_150/meta/norm_stats.json \
  --simulator-assets /path/to/maniskill/task-assets \
  --maniskill-package-assets /path/to/maniskill/package-assets \
  --checkpoint /path/to/runs/maniskill/path-opd-seed0/checkpoints/chunks_000008/checkpoint.pt \
  --checkpoint-sha256 <64-hex-digits> \
  --method path_opd --training-seed 0 \
  --panel /path/to/maniskill/panel_320.json \
  --panel-sha256 <64-hex-digits> \
  --panel-mode custom \
  --output /path/to/evaluations/maniskill/path-opd-seed0
~~~

The evaluator requires the `manifest.json` sidecar emitted beside
`checkpoint.pt`. Before importing CUDA or ManiSkill it verifies the method,
training seed, eight-rank contract, release source hash, RLinf revision, and
all evaluator-visible asset hashes. The frozen-teacher hash remains recorded
but is explicitly reported as unverifiable because evaluation does not need a
teacher path. A raw model state or an old checkpoint without this sidecar is
rejected.

CALVIN ABC-D training and Official-D evaluation also require the scene-D
environment and task-oracle annotation paths. Both commands must be launched
with eight ranks:

~~~bash
torchrun --standalone --nproc_per_node=8 \
  benchmarks/calvin_abc_d/train.py \
  --method path_opd --seed 0 --max-updates 1 \
  --rlinf-checkout /path/to/RLinf \
  --base-model /path/to/calvin/base-model \
  --teacher-model /path/to/calvin/teacher-model \
  --norm /path/to/calvin/norm_stats.json \
  --environment-assets /path/to/calvin/scene-D \
  --task-oracle-annotations /path/to/calvin/evaluation_annotations_and_oracle.yaml \
  --output /path/to/runs/calvin/path-opd-seed0

# Resume that checkpoint through update 2. --max-updates is the absolute
# target update, not the number of additional updates. Use a new output path.
torchrun --standalone --nproc_per_node=8 \
  benchmarks/calvin_abc_d/train.py \
  --method path_opd --seed 0 --max-updates 2 \
  --rlinf-checkout /path/to/RLinf \
  --base-model /path/to/calvin/base-model \
  --teacher-model /path/to/calvin/teacher-model \
  --norm /path/to/calvin/norm_stats.json \
  --environment-assets /path/to/calvin/scene-D \
  --task-oracle-annotations /path/to/calvin/evaluation_annotations_and_oracle.yaml \
  --resume /path/to/runs/calvin/path-opd-seed0/checkpoints/chunks_000008 \
  --output /path/to/runs/calvin/path-opd-seed0-resumed

torchrun --standalone --nproc_per_node=8 \
  benchmarks/calvin_abc_d/evaluate.py \
  --method path_opd --training-seed 0 \
  --rlinf-checkout /path/to/RLinf \
  --base-model /path/to/calvin/base-model \
  --checkpoint /path/to/runs/calvin/path-opd-seed0-resumed/checkpoints/chunks_000016 \
  --norm /path/to/calvin/norm_stats.json \
  --environment-assets /path/to/calvin/scene-D \
  --task-oracle-annotations /path/to/calvin/evaluation_annotations_and_oracle.yaml \
  --panel /path/to/calvin/CALVIN_OFFICIAL_D_SEQUENCES_1000.json \
  --allow-custom-panel \
  --output /path/to/evaluations/calvin/path-opd-seed0
~~~

`--task-oracle-annotations` is a file path, not the CALVIN annotations
directory. Evaluation accepts a YAML/JSON mapping containing both `annotations`
and a Hydra `task_oracle` config, or a raw annotation mapping with the expected
`new_playtable_tasks.yaml` sibling layout. Every panel subtask must have an
annotation in that mapping. The two upstream YAML source files are listed in
[the external-source audit](EXTERNAL_ASSET_SOURCES.md); callers must arrange
them into the accepted layout and record the resulting file identity.

The `--allow-custom-panel` flag is required for a caller-generated panel and
intentionally marks the run as a qualification/custom evaluation. The release
does not ship the upstream Official-D panel, a panel-generation/export script,
or a command that converts the upstream evaluator's generated episodes into
the JSON file consumed here. The upstream protocol is recorded as 1,000
five-subtask sequences, a 360-step subtask limit, and temporary NumPy seed 0,
but those facts alone do not produce or verify this runner's input file. The
example panel path above is therefore a caller-supplied placeholder, not an
artifact generated by this repository.

For a custom qualification panel, `benchmarks/calvin_abc_d/common.py` expects
JSON with a `rows` array; each row has a contiguous zero-based
`sequence_index`, a unique 64-hex `identity_sha256`, an `initial_state` mapping,
and exactly five non-empty `subtasks`. A real eight-rank evaluation requires a
row count from 8 through 1,000 that is divisible by 8; use an eight-row panel
for the smallest distributed qualification. The digest must be recorded with
the run. A panel is considered formal only if it has all 1,000 rows and its
raw-file SHA-256 exactly equals the recorded digest. A generated panel that
does not match that byte digest is accepted only with
`--allow-custom-panel` and must not be reported as an Official-D paper result.
See [the external-source audit](EXTERNAL_ASSET_SOURCES.md) for the upstream
source revisions and dataset routes.

MetaWorld MT50 uses the reconstructed evaluator and an explicit custom-panel
label until a byte-identical public panel is available:

Run this declaration-only command first. It validates the selected task
configuration when available and prints the ordered 500-row panel digest; it
does not import MetaWorld, MuJoCo, CUDA, or OpenPI:

~~~bash
python benchmarks/metaworld_mt50/evaluate.py \
  --rlinf-checkout /path/to/RLinf \
  --checkpoint /path/to/checkpoints/chunks_080000 \
  --norm /path/to/metaworld/norm_stats.json \
  --output artifacts/metaworld-eval-dry-run \
  --method path_opd --training-seed 0 --dry-run
~~~

Copy the reported `panel.actual_sha256` into the real invocation below and
pass it as `--panel-sha256`. A real run fails closed when the supplied digest
does not match the ordered 500-row task panel derived from the selected RLinf
task configuration.

~~~bash
torchrun --standalone --nproc_per_node=8 \
  benchmarks/metaworld_mt50/train.py \
  --method path_opd \
  --rlinf-checkout /path/to/RLinf \
  --base /path/to/metaworld/base-model \
  --teacher /path/to/metaworld/teacher-model \
  --norm /path/to/metaworld/norm_stats.json \
  --output /path/to/runs/metaworld/path-opd-seed0 \
  --seed 0 --max-updates 1

# Continue the saved run through update 2. --max-updates is the absolute
# target count, not the number of additional updates. The checkpoint is read
# from the prior output while results go to a separate directory.
torchrun --standalone --nproc_per_node=8 \
  benchmarks/metaworld_mt50/train.py \
  --method path_opd \
  --rlinf-checkout /path/to/RLinf \
  --base /path/to/metaworld/base-model \
  --teacher /path/to/metaworld/teacher-model \
  --norm /path/to/metaworld/norm_stats.json \
  --output /path/to/runs/metaworld/path-opd-seed0-resumed \
  --seed 0 --resume /path/to/runs/metaworld/path-opd-seed0/checkpoints/chunks_000008 \
  --max-updates 2

python benchmarks/metaworld_mt50/evaluate.py \
  --rlinf-checkout /path/to/RLinf \
  --checkpoint /path/to/runs/metaworld/path-opd-seed0-resumed/checkpoints/chunks_000016 \
  --norm /path/to/metaworld/norm_stats.json \
  --output /path/to/evaluations/metaworld/path-opd-seed0 \
  --method path_opd --training-seed 0 \
  --panel-mode custom --panel-sha256 <64-hex-digits>
~~~

The MetaWorld training/evaluation commands require the caller's pinned
metaworld==3.0.0, MuJoCo, RLinf/OpenPI runtime, and CUDA installation.
The exact external routes and unresolved license/asset entries are listed in
EXTERNAL_ASSET_SOURCES.md; do not replace an unresolved entry with a private
checkpoint or host path in a public report.

## 7. Why paper-scale claims are withheld

The release includes parameterized ManiSkill, MetaWorld, and CALVIN commands
under `benchmarks/`. They validate explicit paths and are intended to perform
real runs when the caller supplies the external runtime and assets. CALVIN
additionally has a dependency-light `--dry-run` contract that checks
configuration, checkpoint, normalization, and environment paths without
importing simulator or CUDA
packages. These entry points are not evidence of a paper-scale rerun: no
eight-GPU or full-panel execution has been completed in this release
environment. CALVIN's real simulator execution, checkpoint restore, and
Official-D evaluator parity remain qualification gates.
The panel and checkpoint identity rules used by those commands are documented
in [Release Contract](RELEASE_CONTRACT.md). In particular, MetaWorld is a
reconstructed evaluator with an explicit custom-panel label, not a formal
paper-panel claim.

The external RLinf checkout and `rlinf-openpi` package are not vendored. For
the runner contract, start from the exact baseline, install the versioned
package, and apply the release patch described in the root README: baseline
commit `fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c`, patch
`integrations/rlinf/path-opd-host-support.patch`, SHA-256
`30296071647c0fcdd36d7b12c08fd5d627b5c2f03e8ac73360def51784b30d6f`, and
`rlinf-openpi==0.1.1`. A different checkout, an unpatched or modified patch,
or a different OpenPI distribution is an unqualified runtime, even if the
Python imports happen to succeed. The real train/evaluate entrypoints now fail
closed before importing CUDA or simulator code when the selected checkout
lacks the pinned revision, patch digest, installed package version, or
patch-created action/continuation interfaces; `--dry-run` and
`--synthetic-smoke` intentionally skip this external gate.
Before any benchmark can be described as paper-exact, all of the following must
be true:

1. Every required asset has an anonymous, reviewer-safe acquisition route,
   exact revision or digest, and verified redistribution/license status.
2. Framework, simulator, model backend, Python, PyTorch, CUDA, and MuJoCo
   versions are pinned and tested together.
3. The ManiSkill and MetaWorld runners pass one-update GPU qualification and
   deterministic regression tests against pinned framework revisions.
4. The CALVIN runner completes a real eight-rank update and checkpoint restore,
   and its evaluator is self-contained and reproduces the saved 1,000-row
   artifacts after removal of its historical external helper dependency.
5. The MetaWorld evaluator is either recovered byte-for-byte or clearly
   labeled as reconstructed and proven equivalent against all saved rows.
6. A clean container completes one-batch training, checkpoint reload, panel
   subset evaluation, and report generation without undeclared host state.

Resource planning for full runs must be measured after this closure. The
historical protocol used eight accelerator ranks, but current evidence does not
support a portable promise for wall time, accelerator memory, disk space, or
specific accelerator models.

## 8. Release audit

Run before packaging:

```bash
python scripts/audit_anonymity.py .
python -m pytest -q
python -m ruff check --no-cache .
```

Then follow the manual checks in [ANONYMITY.md](ANONYMITY.md). Automated success
does not cover hosting-account identity, visitor analytics, archive metadata,
binary metadata, or the legal status of third-party material.

# PVD (Path-wise Velocity Distillation)

This is the anonymous code repository accompanying the PVD submission. PVD
stands for Path-wise Velocity Distillation. It contains the method
implementation, benchmark runners, fixed experiment
contracts, external-asset manifests, tests, and packaging tools. Author names,
institutional paths, project citations, tracking links, model weights,
simulator data, and generated checkpoints are intentionally absent.

PVD supervises a student policy at every pre-transition state on the student's
own sampled flow path. The states are detached from the student's solver
trajectory; a frozen teacher supplies the target velocity in one batched query
per chunk:

```text
student K-step rollout -> K+1 chain states
                       -> discard the endpoint
                       -> detach K pre-transition states
                       -> query the frozen teacher at the same states and times
                       -> crop to the executable action prefix
                       -> mean the K velocity losses
```

The source tree retains `path_opd` in package names, command names, schemas,
configuration keys, and the `--method path_opd` CLI value as stable
implementation identifiers. In the paper and in this README, that
implementation is called PVD. The matched baseline is DAgger and is selected
with the existing `endpoint_dagger` token.

Run all commands below from the repository root. Keep frameworks, datasets,
models, and run outputs outside the Git checkout; the examples use explicit
paths so that no private machine layout is assumed.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/path_opd/core.py` | PVD loss, flow schedule, frozen-teacher guard, action cropping, and trainable-parameter selection |
| `src/path_opd/adapters/openpi.py` | Adapter from RLinf/OpenPI rollout tensors to the PVD core |
| `src/path_opd/adapters/toy.py` | Deterministic CPU example, optimization, checkpoint, and exact resume |
| `src/path_opd/adapters/benchmark_smoke.py` | Synthetic contracts shared by the three benchmark entrypoints |
| `src/path_opd/benchmark.py` | Benchmark configuration loading and external-asset preflight |
| `src/path_opd/assets.py` | Asset manifest and digest validation |
| `src/path_opd/cli.py` | Command-line entry point for smoke tests, benchmark descriptions, and asset preflight |
| `src/path_opd/config.py` | Typed loading and validation of the bundled benchmark contracts |
| `src/path_opd/integrity.py` | Canonical serialization and file/tree digest helpers |
| `src/path_opd/release_contract.py` | Checkpoint, code-provenance, and evaluation-panel contracts |
| `benchmarks/maniskill/{common,train,evaluate}.py` | ManiSkill protocol validation, training, resume, and fixed-panel evaluation |
| `benchmarks/calvin_abc_d/{common,environment,train,evaluate}.py` | CALVIN ABC-D environment adapter, training, resume, and Official-D evaluation |
| `benchmarks/metaworld_mt50/{common,train,evaluate}.py` | MetaWorld MT50 training, resume, and matched 500-row evaluation |
| `benchmarks/{maniskill,calvin_abc_d}/export_panel.py` | Exports fixed evaluation panels from the pinned simulator/evaluator runtime |
| `configs/*.json` | Fixed benchmark protocols used by the runners |
| `configs/assets/*.json` | External model, data, panel, and checkpoint manifests |
| `integrations/rlinf/path-opd-host-support.patch` | Host patch for the pinned RLinf checkout |
| `scripts/benchmark_end_to_end_smoke.py` | Runs all six synthetic train/evaluate entrypoints |
| `scripts/benchmark_environment_smoke.py` | One-step probes for caller-provided simulator installations |
| `scripts/audit_anonymity.py` | Scans the upload tree for common identity leaks |
| `scripts/build_release_archive.py` | Builds a normalized anonymous source archive |
| `docker/Dockerfile` and `docker/entrypoint.sh` | Optional isolated CPU image and its smoke/test entry points |
| `docs/` | Detailed asset provenance, protocol contracts, packaging, and anonymity notes |
| `third_party/rlinf/LICENSE` | License notice for the RLinf integration patch |
| `tests/` | Unit, contract, runner, resume, packaging, and anonymity tests |
| `results/reference/three_benchmark_results.json` | Machine-readable aggregate results associated with the submission |

## 1. Core CPU environment

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are recommended.
The core environment is independent of CUDA and the simulators.

```bash
uv sync --frozen --extra test
.venv/bin/python -m pytest -q

.venv/bin/python -m path_opd.cli smoke \
  --work-dir artifacts/toy-smoke \
  --output artifacts/toy-smoke/report.json

.venv/bin/python -m path_opd.cli benchmark smoke \
  --benchmark all \
  --work-dir artifacts/benchmark-smoke \
  --output artifacts/benchmark-smoke/report.json
```

The public runner boundaries can also be exercised without downloading models
or simulators:

```bash
.venv/bin/python benchmarks/maniskill/train.py --synthetic-smoke
.venv/bin/python benchmarks/calvin_abc_d/train.py --synthetic-smoke
.venv/bin/python benchmarks/metaworld_mt50/train.py --synthetic-smoke

.venv/bin/python benchmarks/maniskill/evaluate.py --synthetic-smoke
.venv/bin/python benchmarks/calvin_abc_d/evaluate.py --synthetic-smoke
.venv/bin/python benchmarks/metaworld_mt50/evaluate.py --synthetic-smoke

.venv/bin/python scripts/benchmark_end_to_end_smoke.py \
  --work-dir artifacts/three-benchmark-e2e-smoke \
  --output artifacts/three-benchmark-e2e-smoke/report.json
```

Do not install the external OpenPI runtime into this `.venv`.
`rlinf-openpi==0.1.1` pins a different PyTorch/CUDA stack, so each real
benchmark receives its own environment below.

## 2. Pinned RLinf runtime

Set explicit source locations:

```bash
export PATH_OPD_REPO="$PWD"
export RLINF_ROOT=/absolute/path/to/RLinf

git clone https://github.com/RLinf/RLinf.git "$RLINF_ROOT"
git -C "$RLINF_ROOT" checkout fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c
test "$(git -C "$RLINF_ROOT" rev-parse HEAD)" = \
  fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c

sha256sum "$PATH_OPD_REPO/integrations/rlinf/path-opd-host-support.patch"
# Expected:
# 30296071647c0fcdd36d7b12c08fd5d627b5c2f03e8ac73360def51784b30d6f

cd "$RLINF_ROOT"
git apply --check "$PATH_OPD_REPO/integrations/rlinf/path-opd-host-support.patch"
git apply "$PATH_OPD_REPO/integrations/rlinf/path-opd-host-support.patch"
```

Apply the patch once. For an already patched checkout,
`git -C "$RLINF_ROOT" apply --reverse --check <patch>` confirms that the
same patch is present.

Create one runtime per simulator using the pinned checkout's installer:

```bash
cd "$RLINF_ROOT"

bash requirements/install.sh embodied \
  --model openpi --env maniskill_libero --venv .venv-path-opd-maniskill

bash requirements/install.sh embodied \
  --model openpi --env calvin --venv .venv-path-opd-calvin

bash requirements/install.sh embodied \
  --model openpi --env metaworld --venv .venv-path-opd-metaworld

cd "$PATH_OPD_REPO"
export PYTHONPATH="$PATH_OPD_REPO/src:$PATH_OPD_REPO${PYTHONPATH:+:$PYTHONPATH}"
```

Each environment must contain `rlinf-openpi==0.1.1`. The relevant external
source pins are:

| Component | Version or revision |
| --- | --- |
| RLinf | `fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c` plus the patch above |
| OpenPI package | `rlinf-openpi==0.1.1` |
| ManiSkill | tag `v3.0.0b22`, commit `33967b9e3ead1f841eec57cc9f31d0d8b8cf0907` |
| MetaWorld | tag `3.0.0`, commit `73c6feed5eda4c5269088e914692e8f295c374e2` |
| CALVIN parent | `fa03f01f19c65920e18cf37398a9ce859274af76` |
| CALVIN environment | `1431a46bd36bde5903fb6345e68b5ccc30def666` |

## 3. Download model and simulator assets

Install the Hugging Face CLI in a small download environment, authenticate if
required, and choose a model directory outside this repository:

```bash
export PATH_OPD_ASSETS=/absolute/path/to/path-opd-assets
mkdir -p "$PATH_OPD_ASSETS"

hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT \
  --revision ffe18938a37d3855c9ff50d6220d39bcd6434f15 \
  --local-dir "$PATH_OPD_ASSETS/calvin-sft"
hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE \
  --revision 712bd3da6f8a92be8004ccf53078af70cc945bc7 \
  --local-dir "$PATH_OPD_ASSETS/calvin-flowsde"

hf download RLinf/RLinf-Pi05-ManiSkill-25Main-SFT \
  --revision eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030 \
  --local-dir "$PATH_OPD_ASSETS/maniskill-sft"
hf download RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE \
  --revision a791196be19a4d35094e1e7eaf4984da3a0ec5ba \
  --local-dir "$PATH_OPD_ASSETS/maniskill-flowsde"

hf download RLinf/RLinf-Pi05-MetaWorld-SFT \
  --revision 60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e \
  --local-dir "$PATH_OPD_ASSETS/metaworld-sft"
hf download RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE \
  --revision 22a30b31e1144a47bf2dc60bcb70c03c5abdf036 \
  --local-dir "$PATH_OPD_ASSETS/metaworld-flowsde"

hf download RLinf/maniskill_assets \
  --repo-type dataset \
  --revision c23fc1880ed7861686d4f995360101eaee4d18a0 \
  --local-dir "$PATH_OPD_ASSETS/maniskill-task-assets"
```

The ManiSkill task bundle above is passed as `--simulator-assets`. It is
different from ManiSkill's package cache (`MS_ASSET_DIR`), which is populated
by the simulator installation and is passed as `--maniskill-package-assets`.

The exact runner paths are:

| Benchmark | Base | Teacher | Normalization statistics |
| --- | --- | --- | --- |
| ManiSkill | `$PATH_OPD_ASSETS/maniskill-sft` | `$PATH_OPD_ASSETS/maniskill-flowsde/actor` | `$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json` |
| CALVIN | `$PATH_OPD_ASSETS/calvin-sft` | `$PATH_OPD_ASSETS/calvin-flowsde` | `$PATH_OPD_ASSETS/calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json` |
| MetaWorld | `$PATH_OPD_ASSETS/metaworld-sft` | `$PATH_OPD_ASSETS/metaworld-flowsde` | `$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json` |

ManiSkill training additionally uses
`$PATH_OPD_ASSETS/maniskill-flowsde/actor/assets/global_step_150/meta/norm_stats.json`
as `--teacher-normalization-stats`.

Verify the runner-selected files:

```bash
cd "$PATH_OPD_ASSETS"
sha256sum -c <<'SHA256'
1ac4fcd76dfe131f9ce4cec94a750424f9eac0275285ebda7ad05ee9550c3bf8  calvin-sft/model.safetensors
0c323860371b91a4835096d3987487367b2b094bf81a22b22812524e462a32aa  calvin-flowsde/model.safetensors
72903e3dad8fc7f4f799e36ee723d0c7f53cb815cc6c55f8ad75f62358f00b41  calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json
9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89  maniskill-sft/model.safetensors
8445f7c3e5dcfa6d42623b2ce975c94280630a12513f2543584afaf84733cbdf  maniskill-flowsde/actor/model.safetensors
67a6e551d2d31140362984b7e7ac7bfe22162545fb498ac803fa38085ed3a089  maniskill-sft/physical-intelligence/maniskill/norm_stats.json
930b816bf2f142f3e377d088d2252e5238b8d082ad8f0abceedfc6296be58852  maniskill-flowsde/actor/assets/global_step_150/meta/norm_stats.json
6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221  metaworld-sft/model.safetensors
e14684cd8e6938ec026eba1d3a11dde757a2b5b503194db85c5b51517fcb11c9  metaworld-flowsde/model.safetensors
ab3e2f7380ccbd0967fe391e07d4bc67492ceb7e3e06a9c03397aa6c28309220  metaworld-sft/lerobot/metaworld_mt50/norm_stats.json
SHA256
cd "$PATH_OPD_REPO"
```

### CALVIN ABC-D data

The official `task_ABC_D.zip` archive is approximately 517 GB. Download it
to a dataset volume, not into this Git repository:

```bash
export PATH_OPD_DATASETS=/absolute/path/to/datasets
mkdir -p "$PATH_OPD_DATASETS/calvin"
cd "$PATH_OPD_DATASETS/calvin"
curl -LO http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
echo "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74  task_ABC_D.zip" | sha256sum -c -
unzip task_ABC_D.zip
cd "$PATH_OPD_REPO"
```

Pin the CALVIN source to the parent and submodule revisions in the table above.
Set `CALVIN_ENV_ASSETS` to the extracted data root. Build
`CALVIN_ORACLE_BUNDLE` from the pinned
`new_playtable_validation.yaml`, `new_playtable_tasks.yaml`, and the
dataset's language annotations. The evaluator accepts a YAML/JSON mapping with
`annotations` and a Hydra `task_oracle` configuration.

## 4. Prepare fixed evaluation inputs

ManiSkill evaluation requires a JSON object with exactly 320 rows,
`environment_id: "PutOnPlateInScene25Main-v3"`, and one unique
`reset_episode_id` per row. The exporter constructs the same pinned RLinf
`ManiskillEnv` used by evaluation and records its real `reset_state_ids` for
eight ranks with 40 environments per rank.

```bash
export MANISKILL_PACKAGE_ASSETS=/absolute/path/to/maniskill-package-cache
export MANISKILL_PANEL=/absolute/path/to/maniskill-panel-320.json

"$RLINF_ROOT/.venv-path-opd-maniskill/bin/python" \
  benchmarks/maniskill/export_panel.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --simulator-assets "$PATH_OPD_ASSETS/maniskill-task-assets" \
  --maniskill-package-assets "$MANISKILL_PACKAGE_ASSETS" \
  --output "$MANISKILL_PANEL"

export MANISKILL_PANEL_SHA256=$(sha256sum "$MANISKILL_PANEL" | cut -d' ' -f1)
```

CALVIN evaluation uses the pinned upstream sequence generator. The exporter
calls `get_sequences(1000)`, preserves every real initial state and five-task
sequence, assigns stable identities, and writes both the panel JSON and a
sidecar JSONL file.

```bash
export CALVIN_PANEL=/absolute/path/to/calvin-official-d-panel.json

"$RLINF_ROOT/.venv-path-opd-calvin/bin/python" \
  benchmarks/calvin_abc_d/export_panel.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --output "$CALVIN_PANEL"
```

MetaWorld needs no panel file. Its evaluator deterministically builds 50 tasks
times 10 variants from the pinned RLinf task configuration.

## 5. ManiSkill

Define the remaining external path:

```bash
export MS_TORCHRUN="$RLINF_ROOT/.venv-path-opd-maniskill/bin/torchrun"
```

Inspect the resolved one-update plan by appending `--dry-run` to the training
command. Run one update on eight processes:

```bash
"$MS_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/maniskill/train.py \
  --method path_opd --seed 0 --world-size 8 --max-updates 1 \
  --rlinf-root "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/maniskill-sft" \
  --teacher-model "$PATH_OPD_ASSETS/maniskill-flowsde/actor" \
  --normalization-stats "$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json" \
  --teacher-normalization-stats "$PATH_OPD_ASSETS/maniskill-flowsde/actor/assets/global_step_150/meta/norm_stats.json" \
  --simulator-assets "$PATH_OPD_ASSETS/maniskill-task-assets" \
  --maniskill-package-assets "$MANISKILL_PACKAGE_ASSETS" \
  --output-dir artifacts/maniskill-seed0
```

Resume to the absolute target of two updates. ManiSkill resume takes the
`checkpoint.pt` file:

```bash
"$MS_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/maniskill/train.py \
  --method path_opd --seed 0 --world-size 8 --max-updates 2 \
  --rlinf-root "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/maniskill-sft" \
  --teacher-model "$PATH_OPD_ASSETS/maniskill-flowsde/actor" \
  --normalization-stats "$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json" \
  --teacher-normalization-stats "$PATH_OPD_ASSETS/maniskill-flowsde/actor/assets/global_step_150/meta/norm_stats.json" \
  --simulator-assets "$PATH_OPD_ASSETS/maniskill-task-assets" \
  --maniskill-package-assets "$MANISKILL_PACKAGE_ASSETS" \
  --output-dir artifacts/maniskill-seed0 \
  --resume artifacts/maniskill-seed0/checkpoints/chunks_000008/checkpoint.pt
```

Evaluate a saved checkpoint:

```bash
export MANISKILL_CHECKPOINT=artifacts/maniskill-seed0/checkpoints/chunks_000016/checkpoint.pt
export MANISKILL_CHECKPOINT_SHA256=$(sha256sum "$MANISKILL_CHECKPOINT" | cut -d' ' -f1)

"$MS_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/maniskill/evaluate.py \
  --rlinf-root "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/maniskill-sft" \
  --normalization-stats "$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json" \
  --simulator-assets "$PATH_OPD_ASSETS/maniskill-task-assets" \
  --maniskill-package-assets "$MANISKILL_PACKAGE_ASSETS" \
  --checkpoint "$MANISKILL_CHECKPOINT" \
  --checkpoint-sha256 "$MANISKILL_CHECKPOINT_SHA256" \
  --method path_opd --training-seed 0 \
  --panel "$MANISKILL_PANEL" --panel-mode custom \
  --panel-sha256 "$MANISKILL_PANEL_SHA256" \
  --output artifacts/maniskill-seed0/evaluation
```

## 6. CALVIN ABC-D

```bash
export CALVIN_TORCHRUN="$RLINF_ROOT/.venv-path-opd-calvin/bin/torchrun"

"$CALVIN_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/calvin_abc_d/train.py \
  --method path_opd --seed 0 --world-size 8 --max-updates 1 \
  --rlinf-checkout "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/calvin-sft" \
  --teacher-model "$PATH_OPD_ASSETS/calvin-flowsde" \
  --normalization-stats "$PATH_OPD_ASSETS/calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json" \
  --environment-assets "$CALVIN_ENV_ASSETS" \
  --task-oracle-annotations "$CALVIN_ORACLE_BUNDLE" \
  --output artifacts/calvin-seed0
```

Resume to two updates. CALVIN accepts the checkpoint directory:

```bash
"$CALVIN_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/calvin_abc_d/train.py \
  --method path_opd --seed 0 --world-size 8 --max-updates 2 \
  --rlinf-checkout "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/calvin-sft" \
  --teacher-model "$PATH_OPD_ASSETS/calvin-flowsde" \
  --normalization-stats "$PATH_OPD_ASSETS/calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json" \
  --environment-assets "$CALVIN_ENV_ASSETS" \
  --task-oracle-annotations "$CALVIN_ORACLE_BUNDLE" \
  --output artifacts/calvin-seed0 \
  --resume artifacts/calvin-seed0/checkpoints/chunks_000008
```

```bash
"$CALVIN_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/calvin_abc_d/evaluate.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --base-model "$PATH_OPD_ASSETS/calvin-sft" \
  --checkpoint artifacts/calvin-seed0/checkpoints/chunks_000016 \
  --normalization-stats "$PATH_OPD_ASSETS/calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json" \
  --environment-assets "$CALVIN_ENV_ASSETS" \
  --task-oracle-annotations "$CALVIN_ORACLE_BUNDLE" \
  --panel "$CALVIN_PANEL" --allow-custom-panel \
  --method path_opd --training-seed 0 \
  --output artifacts/calvin-seed0/evaluation
```

Omit `--allow-custom-panel` only when the panel is the exact 1,000-row
Official-D file with SHA-256
`0aaa7c37dd5976b3f4372b8501e67e692ff565ba48730c0481aa3f92ff71ff60`.

## 7. MetaWorld MT50

First inspect both plans. The evaluation plan prints
`panel.actual_sha256`; save that value for the real evaluation.

```bash
export MW_PY="$RLINF_ROOT/.venv-path-opd-metaworld/bin/python"
export MW_TORCHRUN="$RLINF_ROOT/.venv-path-opd-metaworld/bin/torchrun"

"$MW_PY" benchmarks/metaworld_mt50/train.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --base "$PATH_OPD_ASSETS/metaworld-sft" \
  --teacher "$PATH_OPD_ASSETS/metaworld-flowsde" \
  --norm "$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json" \
  --output artifacts/metaworld-dry-run/train \
  --method path_opd --seed 0 --max-updates 1 --dry-run

"$MW_PY" benchmarks/metaworld_mt50/evaluate.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --checkpoint /path/to/checkpoints/chunks_080000 \
  --norm "$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json" \
  --output artifacts/metaworld-dry-run/evaluate \
  --method path_opd --training-seed 0 --dry-run
```

Train once and resume to an absolute target of two updates. MetaWorld resume
takes the directory containing `trainer_state.pt` and `manifest.json`.

```bash
"$MW_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/metaworld_mt50/train.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --base "$PATH_OPD_ASSETS/metaworld-sft" \
  --teacher "$PATH_OPD_ASSETS/metaworld-flowsde" \
  --norm "$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json" \
  --output artifacts/metaworld-seed0 \
  --method path_opd --seed 0 --max-updates 1

"$MW_TORCHRUN" --standalone --nproc-per-node=8 benchmarks/metaworld_mt50/train.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --base "$PATH_OPD_ASSETS/metaworld-sft" \
  --teacher "$PATH_OPD_ASSETS/metaworld-flowsde" \
  --norm "$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json" \
  --output artifacts/metaworld-seed0 \
  --method path_opd --seed 0 --max-updates 2 \
  --resume artifacts/metaworld-seed0/checkpoints/chunks_000008
```

```bash
export METAWORLD_PANEL_SHA256=<panel.actual_sha256-from-dry-run>

"$MW_PY" benchmarks/metaworld_mt50/evaluate.py \
  --rlinf-checkout "$RLINF_ROOT" \
  --checkpoint artifacts/metaworld-seed0/checkpoints/chunks_000016 \
  --norm "$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json" \
  --output artifacts/metaworld-seed0/evaluation \
  --method path_opd --training-seed 0 \
  --panel-mode custom --panel-sha256 "$METAWORLD_PANEL_SHA256"
```

Repeat the complete training and evaluation procedure for ManiSkill and CALVIN
seeds `0,1,2`, MetaWorld seeds `0,1`, and both PVD (`path_opd`) and DAgger
(`endpoint_dagger`). Full runs use the default 80,000 chunks and 10,000
updates; `--max-updates` is an absolute target, not an additional count.

## 8. Outputs and release checks

Each run writes checkpoints below `<output>/checkpoints/chunks_XXXXXX/` and
evaluation rows/summaries below its evaluation output. Keep these generated
directories outside the source archive when they contain large weights.

Before publishing an anonymous archive:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check --no-cache .
.venv/bin/python -m compileall -q src benchmarks tests scripts
.venv/bin/python scripts/audit_anonymity.py .

.venv/bin/python scripts/build_release_archive.py \
  --root "$PWD" \
  --output /tmp/pvd-anonymous-review.tar
```

The archive builder excludes ignored generated state, rejects model and
checkpoint binaries, normalizes ownership and timestamps, and runs the
anonymity scanner before writing the tar file. Third-party source, model,
dataset, and simulator licenses remain applicable to assets downloaded by the
commands above.

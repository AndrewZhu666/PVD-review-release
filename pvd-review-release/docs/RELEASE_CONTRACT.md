# Release Contract

The benchmark scripts distinguish an executable run from a paper-exact claim.
Every real run records the input bytes, the release source tree, the selected
RLinf checkout revision when available, and the panel identity.  A resume is
rejected when any of those identities changes.

## ManiSkill

The public asset manifest contains the expected SHA-256 for the 320-row panel.
Use `--panel-mode formal` (the default) only with that exact file and pass its
observed SHA-256:

```bash
PANEL_SHA256="$(sha256sum /path/to/panel_320.json | awk '{print $1}')"
torchrun --standalone --nproc-per-node=8 \
  benchmarks/maniskill/evaluate.py \
  --rlinf-root /path/to/RLinf \
  --base-model /path/to/base-model \
  --normalization-stats /path/to/norm.json \
  --simulator-assets /path/to/rlinf-assets \
  --maniskill-package-assets /path/to/mani-skill-assets \
  --checkpoint /path/to/checkpoint.pt \
  --checkpoint-sha256 "$(sha256sum /path/to/checkpoint.pt | awk '{print $1}')" \
  --method path_opd --training-seed 0 \
  --panel /path/to/panel_320.json \
  --panel-sha256 "$PANEL_SHA256" \
  --panel-mode formal \
  --output artifacts/maniskill-eval.json
```

If the panel is caller-generated, pass `--panel-mode custom` and its exact
digest.  The output then carries `formal: false` and must not be reported as a
formal paper-panel result.  The preflight is not a substitute for execution;
the panel and checkpoint are hashed again after the distributed workers start.

For a CPU-only contract check, add `--dry-run`.  It does not import CUDA,
RLinf, OpenPI, or ManiSkill and does not produce benchmark metrics.

Training checkpoints contain a `release_contract` object in both
`checkpoint.pt` and the sidecar `manifest.json`. It binds the method, seed,
world size, fixed protocol, base model, teacher, normalizer, simulator assets,
ManiSkill assets, release source tree, and RLinf revision. Evaluation requires
`--method` and `--training-seed` to match that sidecar and rejects a raw model
state or an old checkpoint without it before importing CUDA. `--resume` fails
closed if any binding differs.

## CALVIN ABC-D

CALVIN training uses the ABC task-suite wrapper, while Official-D evaluation
uses the raw scene so the fixed panel can provide explicit reset states. New
checkpoints store one environment continuation and RNG state per rank. A
legacy rank-zero-only checkpoint is rejected for the eight-rank resume path;
the runner will not silently restore all workers to the same simulator state.

## MetaWorld MT50

This release does not contain a public, fixed 500-row MetaWorld panel digest.
The evaluator is explicitly labelled `reconstructed_evaluator`, and its
default panel mode is `custom`.  A dry-run computes the generated panel
identity from the ordered task/prompt configuration and protocol:

```bash
python benchmarks/metaworld_mt50/evaluate.py \
  --rlinf-checkout /path/to/RLinf \
  --checkpoint /path/to/checkpoint \
  --norm /path/to/norm_stats.json \
  --output artifacts/metaworld-eval \
  --method path_opd --training-seed 0 --dry-run
```

For a real run, pass the printed digest explicitly as
`--panel-sha256 <digest>`.  The result remains custom even when the digest
matches a prior run.  `--panel-mode formal` is intentionally rejected because
the release manifest has no published formal digest.

## What can be claimed

The core tests and dry-runs establish the executable contract and dependency
boundaries.  A one-update GPU run is a qualification run.  A full 80,000-chunk
training run and a complete fixed-panel evaluation may be described as a
reproduction run only after the supplied external assets, framework versions,
and generated reports are archived with their hashes.

The repository records public routes for the ManiSkill and MetaWorld model
names, the ManiSkill simulator bundle, and selected CALVIN model/statistics
entries. Immutable revisions and model-file hashes for the ManiSkill and
MetaWorld models were recovered from official download-cache metadata; live
Hub/API and license confirmation remain pending. The historical environments,
panels, annotations, and checkpoints remain unresolved. These entries are not
an automatic downloader and do not grant blanket redistribution rights. Until
the remaining inputs are published with exact revisions and licenses, do not
claim that the current tree alone reproduces the paper numbers. The historical
reference JSON is sealed evidence, not a current-release rerun.

# External asset and framework sources

This note is the source-of-truth index for files that are intentionally **not**
included in this review release.  It records canonical first-party URLs, the
revision that was observed, the path that a runner consumes, and the license
metadata exposed by the source.  It does not grant permission to mirror or
redistribute any model, dataset, simulator, checkpoint, or evaluation panel.

**Audit date:** 2026-09-23.  A revision is a Git commit, tag, or
package artifact only when the text below says so explicitly.  A model-file
SHA-256 is a digest of that file; it is not a hash of the whole model
directory.

## Status vocabulary

- **Verified** means that the first-party repository/package metadata and the
  referenced revision agree.
- **Route identified** means that an official download page and a concrete
  route are known, but one or more license, tree-digest, or file-level details
  remain unresolved.
- **Unresolved** is deliberate: the release must fail closed rather than turn
  an absent URL, revision, or license into an implicit claim.

The canonical Hugging Face URLs are listed below.  If a client cannot retrieve
an endpoint at a given time, it should retry the canonical URL.  This audit
also records immutable revisions and file hashes recovered from the official
Hugging Face download-cache metadata on the qualification host.  That cache is
first-party download evidence, but it is not a substitute for a live Hub API
check; entries marked pending still require that revalidation before a
paper-scale claim.

## At-a-glance routes

| Component | Canonical first-party source | Pinned revision/artifact | Consumed paths | License evidence | Status |
| --- | --- | --- | --- | --- | --- |
| RLinf runner | [RLinf commit](https://github.com/RLinf/RLinf/commit/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c) | Git commit `fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c` | `requirements/install.sh`; embodied runner/config files | Apache-2.0 `LICENSE` at that commit | Verified |
| RLinf OpenPI runtime | [PyPI `rlinf-openpi` 0.1.1](https://pypi.org/project/rlinf-openpi/0.1.1/) | PyPI wheel `rlinf_openpi-0.1.1-py3-none-any.whl`; SHA-256 `e7b8e6e61d575487bce63e89dee5e7af7834904ee07152bcbbd5b58c06e4c47d` | Installed `openpi/` package | Apache-2.0 in wheel metadata; project URL points to [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Verified as an artifact; no VCS commit published for the wheel |
| CALVIN π0.5 SFT | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT) | model revision `ffe18938a37d3855c9ff50d6220d39bcd6434f15` | `model.safetensors`; nested `InternRobotics/InternData-Calvin_ABC/norm_stats.json` | HF card tag `apache-2.0` | Route identified |
| CALVIN π0.5 FlowSDE teacher | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE) | model revision `712bd3da6f8a92be8004ccf53078af70cc945bc7` | `model.safetensors`; nested normalization file is present in the model tree | No license card/tag was exposed by the observed metadata | Route identified; license unresolved |
| ManiSkill package | [ManiSkill](https://github.com/mani-skill/ManiSkill) | runner baseline tag `v3.0.0b22`, commit `33967b9e3ead1f841eec57cc9f31d0d8b8cf0907` | simulator package and its `MS_ASSET_DIR` cache | Repository Apache-2.0; bundled-asset terms require separate review | Verified runner source pin; asset terms remain per-file |
| RLinf ManiSkill task assets | [HF dataset](https://huggingface.co/datasets/RLinf/maniskill_assets) | dataset revision `c23fc1880ed7861686d4f995360101eaee4d18a0` | RLinf task-asset directory named `assets` | Dataset card says ODC-By v1.0 for the collection and points to Objaverse-XL; individual objects have different terms | Route identified; per-object license review required |
| ManiSkill π0.5 SFT | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT) | revision `eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030` | `model.safetensors` (`9877be...`, 7,473,091,464 bytes); `physical-intelligence/maniskill/norm_stats.json` | Local official card metadata says MIT; live card/API revalidation pending | Route and file identity recorded; live license/API revalidation pending |
| ManiSkill FlowSDE teacher | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE) | revision `a791196be19a4d35094e1e7eaf4984da3a0ec5ba` | `actor/model.safetensors` (`8445f7...`, 8,537,405,116 bytes); `actor/assets/global_step_150/meta/norm_stats.json` | Local official card metadata says MIT; live card/API revalidation pending | Route and file identity recorded; live license/API revalidation pending |
| MetaWorld | [MetaWorld tag](https://github.com/Farama-Foundation/Metaworld/tree/3.0.0) | tag `3.0.0` (without `v`), commit `73c6feed5eda4c5269088e914692e8f295c374e2` | Python package and MuJoCo XML assets shipped by MetaWorld | MIT `LICENSE` at the tag | Verified |
| MetaWorld π0.5 SFT | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT) | revision `60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e` | `model.safetensors` (`6e877a...`, 7,473,091,464 bytes); `lerobot/metaworld_mt50/norm_stats.json` (`ab3e2f...`) | No card license in the observed official cache | Route and file identity recorded; license/API revalidation pending |
| MetaWorld FlowSDE teacher | [HF model](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE) | revision `22a30b31e1144a47bf2dc60bcb70c03c5abdf036` | `model.safetensors` (`e14684...`, 8,531,240,580 bytes); `lerobot/metaworld_mt50/norm_stats.json` (`ab3e2f...`) | No card license in the observed official cache | Route and file identity recorded; license/API revalidation pending |
| CALVIN software | [CALVIN parent](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76) and [calvin_env submodule](https://github.com/mees/calvin_env/tree/1431a46bd36bde5903fb6345e68b5ccc30def666) | parent `fa03f01f...`; submodule `1431a46...` | CALVIN code, environment, task oracle, and annotations | MIT in both source repositories; dataset license not established | Verified source pins; data/panel closure unresolved |

## RLinf and the OpenPI boundary

### RLinf

The pinned RLinf source object is the signed GitHub commit
[`fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c`](https://api.github.com/repos/RLinf/RLinf/commits/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c).
Its commit message is the runner synchronization fix used by the integration
route.  The upstream license is the Apache License 2.0 at
[`LICENSE`](https://raw.githubusercontent.com/RLinf/RLinf/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/LICENSE).
The commit's official configuration files identify the external OpenPI
dependency and contain the three benchmark configuration families:

- [`requirements/install.sh`](https://github.com/RLinf/RLinf/blob/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/requirements/install.sh)
  is the actual environment entrypoint. Its OpenPI routes install
  `rlinf-openpi==0.1.1`; its ManiSkill route installs
  `ManiSkill.git@v3.0.0b22`; and its MetaWorld route installs
  `metaworld==3.0.0`.
- [`requirements/embodied/models/openpi.txt`](https://github.com/RLinf/RLinf/blob/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/requirements/embodied/models/openpi.txt)
  records the JAX/Orbax compatibility pins used by that runtime.
- The official [CALVIN](https://github.com/RLinf/RLinf/blob/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/examples/embodiment/config/calvin_abc_d_ppo_openpi_pi05.yaml),
  [ManiSkill](https://github.com/RLinf/RLinf/blob/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml),
  and [MetaWorld](https://github.com/RLinf/RLinf/blob/fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c/examples/embodiment/config/metaworld_50_ppo_openpi_pi05.yaml)
  YAML files show the upstream model/config boundary.  They are reference
  configurations, not a license for the associated weights.

The MetaWorld runner also consumes the ordered task/prompt mapping
`metaworld_config.json` from the selected RLinf checkout. The supported paths
are `rlinf/envs/metaworld/metaworld_config.json` and, for the alternate layout
used by some pinned source trees, `rlinf/envs/sim/metaworld/metaworld_config.json`.
The file is framework-owned and is therefore not duplicated in the asset
manifest; its bytes are hashed into the MetaWorld checkpoint and panel
contracts. A checkout with neither path is rejected before importing the
simulator. The source and license are the pinned RLinf commit and its
Apache-2.0 license above; no separate private copy is expected.

The SHA-256 recorded for RLinf in `configs/provenance.json` has
`sha256_scope=integration_patch`: it is the release's small integration patch,
not a digest of the RLinf source tree.  Do not compare those two objects as if
they were the same checkout.

### OpenPI: package versus Git repository

There are two names that must not be conflated:

1. The runner's install script requests the **PyPI distribution**
   `rlinf-openpi==0.1.1`.  The official PyPI JSON endpoint is
   [`pypi.org/pypi/rlinf-openpi/0.1.1/json`](https://pypi.org/pypi/rlinf-openpi/0.1.1/json).
   Its metadata says Apache-2.0 and gives the project URL
   [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi).
   The wheel used for the recorded digest is linked from that API response and
   has SHA-256
   `e7b8e6e61d575487bce63e89dee5e7af7834904ee07152bcbbd5b58c06e4c47d`.
2. [`RLinf/openpi`](https://github.com/RLinf/openpi) is a public GitHub fork,
   but the official repository refs do not expose a Git tag or commit named
   `0.1.1`.  Therefore `0.1.1` in this release means a **package version**, not
   a Git revision.  The wheel digest, rather than a fabricated Git SHA, is the
   reproducibility pin.

The PyPI artifact includes the upstream Apache-2.0 license and a separate
Gemma license notice.  Anyone redistributing the package must retain both
notices and comply with the package's dependency licenses.  Its published
metadata also declares `torch==2.7.1` (plus CUDA/JAX and simulator
dependencies); this differs from the release's CPU-core lock and is why the
README requires a separate benchmark runtime environment.

Do not approximate the external runtime with only `pip install
rlinf-openpi==0.1.1`. From the pinned RLinf root, provision each benchmark in
its own environment through the existing installer:

```bash
bash requirements/install.sh embodied \
  --model openpi --env maniskill_libero --venv .venv-path-opd-maniskill
bash requirements/install.sh embodied \
  --model openpi --env metaworld --venv .venv-path-opd-metaworld
bash requirements/install.sh embodied \
  --model openpi --env calvin --venv .venv-path-opd-calvin
```

These commands can download packages, source repositories, and package assets.
The exact clone, commit check, host-patch, and environment procedure is in the
RLinf setup instructions in the root README. The CALVIN branch of
the installer does not pin the CALVIN checkout, so its observed parent and
submodule revisions still have to be recorded separately for a qualified run.

## Fixed-revision model acquisition

The six model snapshots below are the public base/teacher routes recorded by
the manifests. The commands deliberately pin immutable revisions and preserve
repository-relative paths. They download complete model snapshots because the
external OpenPI loader may consume repository metadata in addition to the
weight file. They are large downloads and were not executed by the release's
CPU tests.

Run from any caller-owned directory after installing the official Hugging Face
CLI, and replace the absolute destination before copying the commands:

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
```

The runner arguments must point at the following exact directories/files. In
particular, the ManiSkill teacher is the nested `actor/` directory, not its
download root.

| Benchmark | Base model directory | Teacher model directory | Normalizer file |
| --- | --- | --- | --- |
| CALVIN | `$PATH_OPD_ASSETS/calvin-sft` | `$PATH_OPD_ASSETS/calvin-flowsde` | `$PATH_OPD_ASSETS/calvin-sft/InternRobotics/InternData-Calvin_ABC/norm_stats.json` |
| ManiSkill | `$PATH_OPD_ASSETS/maniskill-sft` | `$PATH_OPD_ASSETS/maniskill-flowsde/actor` | `$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json` |
| MetaWorld | `$PATH_OPD_ASSETS/metaworld-sft` | `$PATH_OPD_ASSETS/metaworld-flowsde` | `$PATH_OPD_ASSETS/metaworld-sft/lerobot/metaworld_mt50/norm_stats.json` |

From the asset root, this verifies every runner-selected weight and normalizer
as raw downloaded bytes:

```bash
(
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
)
```

There are two different hash scopes for the ManiSkill normalizer, and both are
intentional:

- `67a6e551...` above is the raw-file SHA-256 produced by `sha256sum`.
- `229cea8f...` in `configs/assets/maniskill.json` uses
  `sha256_canonical_json`: decode the JSON, serialize with ASCII escapes,
  sorted keys, and compact separators, then hash those canonical bytes. It is
  stable across whitespace-only JSON changes and is therefore not expected to
  equal the raw-file digest.

From the release root, the exact manifest-compatible check is:

```bash
export PATH_OPD_RELEASE=/absolute/path/to/path-opd-review
maniskill_norm="$PATH_OPD_ASSETS/maniskill-sft/physical-intelligence/maniskill/norm_stats.json"
canonical_digest="$(PYTHONPATH="$PATH_OPD_RELEASE/src" python -c \
  'import sys; from path_opd.assets import sha256_canonical_json_file as digest; print(digest(sys.argv[1]))' \
  "$maniskill_norm")"
test "$canonical_digest" = \
  229cea8fece49762eb8c940e95c0e33b62a3f0473ee9085a9e519cdab6dccdd5
```

All other digests in the raw-byte block use the manifest's `sha256_file`
semantics. Passing these checks establishes file identity only. It does not
resolve pending license/API checks, simulator assets, formal panels, paper
checkpoints, or real-runtime qualification; the overall paper-reproduction
status remains `NOT_QUALIFIED`.

## CALVIN ABC-D model and normalization routes

### SFT initialization

The public SFT model card is
[`RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT).
The observed immutable revision is
`ffe18938a37d3855c9ff50d6220d39bcd6434f15`.  At that revision the model tree
contains:

| File | Canonical URL pattern | Recorded SHA-256/size |
| --- | --- | --- |
| `model.safetensors` | [`resolve/ffe.../model.safetensors`](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT/resolve/ffe18938a37d3855c9ff50d6220d39bcd6434f15/model.safetensors) | `1ac4fcd76dfe131f9ce4cec94a750424f9eac0275285ebda7ad05ee9550c3bf8` (about 7.47 GB) |
| `InternRobotics/InternData-Calvin_ABC/norm_stats.json` | [`blob/ffe.../InternRobotics/InternData-Calvin_ABC/norm_stats.json`](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT/blob/ffe18938a37d3855c9ff50d6220d39bcd6434f15/InternRobotics/InternData-Calvin_ABC/norm_stats.json) | `72903e3dad8fc7f4f799e36ee723d0c7f53cb815cc6c55f8ad75f62358f00b41` (1,755 bytes) |

The exact nested path is significant: downloading only the repository root or
looking for a top-level `norm_stats.json` is not equivalent.  A download that
does not preserve this path must pass the resulting file explicitly through
`calvin_normalization_stats`.

The corresponding official API endpoints are the [model metadata](https://huggingface.co/api/models/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT)
and [revision tree](https://huggingface.co/api/models/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT/tree/ffe18938a37d3855c9ff50d6220d39bcd6434f15?expand=1).
The card metadata advertises `apache-2.0`; this release records that fact for
the SFT model and the nested normalization file.  The model itself is not
redistributed here.

The fixed-revision command, raw-file digest checks, and exact local argument
paths are consolidated in [Fixed-revision model acquisition](#fixed-revision-model-acquisition).

### Frozen FlowSDE teacher

The teacher card is
[`RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE`](https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE)
at revision `712bd3da6f8a92be8004ccf53078af70cc945bc7`.  The recorded
`model.safetensors` SHA-256 is
`0c323860371b91a4835096d3987487367b2b094bf81a22b22812524e462a32aa`
(about 8.53 GB).  The tree also contains the same 1,755-byte normalization
file under `InternRobotics/InternData-Calvin_ABC/norm_stats.json` (and a
duplicated `InternRobotics/InternRobotics/...` path).  Use the single path named
in the SFT section for this release.

Unlike the SFT card, the observed teacher metadata had neither a license tag
nor card-level license field.  Its manifest entry therefore remains
`license=unknown`; do not infer Apache-2.0 from the neighboring SFT model.

### CALVIN simulator and annotations

The upstream CALVIN software repository is
[`mees/calvin`](https://github.com/mees/calvin), whose repository metadata and
`LICENSE` identify MIT.  The simulator source can be pinned to parent commit
[`fa03f01f19c65920e18cf37398a9ce859274af76`](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76)
with `calvin_env` submodule commit
[`1431a46bd36bde5903fb6345e68b5ccc30def666`](https://github.com/mees/calvin_env/tree/1431a46bd36bde5903fb6345e68b5ccc30def666).
That software pin does not close the scene-D data, annotations, generated
Official-D panel, or paper checkpoints, so the corresponding manifest entries
intentionally remain unknown.  A real run must provide those inputs and record
their own revision, digest, and license before claiming paper-level
reproduction.

The official ABC-D archive is
[`task_ABC_D.zip`](http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip),
advertised at about 517 GB, with SHA-256
`c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74` in the
official [`sha256sum.txt`](http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt).
The route is HTTP-only in the observed source, while the release manifest
requires HTTPS URLs, so it is documented here but deliberately not inserted as
a `public_url`.  The archive contains language annotations under
`train/validation/lang_annotations/auto_lang_ann.npy`; the task-oracle files
are separate pinned-source files under
`calvin_models/conf/annotations/new_playtable_validation.yaml` and
`calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml`.

Official-D is generated by the pinned evaluator (`NUM_SEQUENCES=1000`,
`EP_LEN=360`, temporary NumPy seed `0`) rather than distributed as a panel
file.  The release's known panel digest is therefore an expected historical
identity, not a public download route; it stays fail-closed until generated
rows are verified against the exact source/runtime.

For the simulator boundary only, the upstream repository records a public
`calvin_env` submodule.  One observed parent/submodule pair is the public
parent commit [`fa03f01f19c65920e18cf37398a9ce859274af76`](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76)
and the submodule commit
[`1431a46bd36bde5903fb6345e68b5ccc30def666`](https://github.com/mees/calvin_env/tree/1431a46bd36bde5903fb6345e68b5ccc30def666).
The corresponding checkout exposes the simulator data root consumed by the
probe.  These revisions are a useful starting point, not a complete CALVIN
release pin: the task annotations, Official-D panel, dataset provenance, and
license/file-level closure remain unresolved in this snapshot.

## ManiSkill package and task assets

The canonical ManiSkill repository is now
[`mani-skill/ManiSkill`](https://github.com/mani-skill/ManiSkill) (the historical
`haosulab/ManiSkill` URL redirects). The actual runner baseline is tag
[`v3.0.0b22`](https://github.com/mani-skill/ManiSkill/tree/v3.0.0b22), commit
`33967b9e3ead1f841eec57cc9f31d0d8b8cf0907`, because that is what the pinned
RLinf installer selects. Tag `v3.0.1`, commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`, is retained only as a research
reference for later upstream asset metadata; it is not compatible evidence for
the runner baseline. The repository `LICENSE` is Apache-2.0. The official
v3.0.1 README separately states that rigid-body environments use permissive
licenses while bundled assets are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
This reference is useful for license review, but the exact v3.0.0b22 assets
still require their own review when the `MS_ASSET_DIR` cache is populated.

RLinf's task-asset bundle is the first-party dataset
[`RLinf/maniskill_assets`](https://huggingface.co/datasets/RLinf/maniskill_assets)
at revision `c23fc1880ed7861686d4f995360101eaee4d18a0`.  Its dataset card
provides the download command and says to place the result in an RLinf
ManiSkill `assets` directory.  The card labels the **collection** ODC-By v1.0,
attributes the objects to [Objaverse-XL](https://huggingface.co/datasets/allenai/objaverse-xl),
and warns that individual Objaverse objects carry different licenses.  Thus
`ODC-By` in the manifest is collection-level metadata, not a blanket license
for every mesh, texture, or derived object.  No complete tree SHA-256 is
published in this release.

The official dataset metadata endpoint is
[`api/datasets/RLinf/maniskill_assets`](https://huggingface.co/api/datasets/RLinf/maniskill_assets);
the immutable tree route is
[`tree/c23fc...`](https://huggingface.co/datasets/RLinf/maniskill_assets/tree/c23fc1880ed7861686d4f995360101eaee4d18a0).
An acquisition command that preserves the fixed dataset revision is:

```bash
hf download RLinf/maniskill_assets \
  --repo-type dataset \
  --revision c23fc1880ed7861686d4f995360101eaee4d18a0 \
  --local-dir "$PATH_OPD_ASSETS/maniskill-task-assets"
```

Use the resulting directory as the runner's `--simulator-assets` input. This
command identifies the dataset snapshot but does not provide a complete tree
digest or per-object license clearance; it cannot by itself close the
ManiSkill qualification gate.

The package's own asset downloader also references additional task, scene, and
robot sources.  Those are not silently folded into the RLinf dataset route:
each source must be pinned and license-reviewed if a selected task needs it.

As a research reference, ManiSkill v3.0.1 also names a Bridge-v2 asset archive at
[`bridge_v2_real2sim_dataset.zip`](https://huggingface.co/datasets/haosulab/ManiSkill_bridge_v2_real2sim/resolve/main/bridge_v2_real2sim_dataset.zip)
with SHA-256
`618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92`.
The URL uses the mutable `main` revision and the archive is described as
CC BY-NC 4.0, so it is a useful acquisition/checksum hint, not an immutable
manifest route or a replacement for the v3.0.0b22 runner baseline; the JSON
asset contract remains unresolved for it.

The release distinguishes two caller inputs:

- `maniskill_simulator_assets`: the RLinf task-asset bundle above;
- `maniskill_package_assets`: the ManiSkill package cache used through
  `MS_ASSET_DIR`.

The latter has no single public route or tree digest in this snapshot and stays
unresolved.  Supplying one directory for both parameters is not evidence that
the two asset sets are complete.

The ManiSkill SFT and FlowSDE model routes are now identified from the official
RLinf model names.  Immutable revisions and model-file digests were recovered
from official Hugging Face download-cache metadata, and the SFT route contains
the normalizer path recorded in the manifest.  The cache is useful download
evidence, but live Hub/API and card revalidation is still pending; these
entries remain caller-supplied and must not be treated as redistributable
model artifacts.

## MetaWorld MT50 and MuJoCo

The official MetaWorld release used by the upstream install route is tag
`3.0.0` (the tag is intentionally **without** a leading `v`).  It resolves to
commit `73c6feed5eda4c5269088e914692e8f295c374e2`:

- [tag tree](https://github.com/Farama-Foundation/Metaworld/tree/3.0.0)
- [commit](https://github.com/Farama-Foundation/Metaworld/commit/73c6feed5eda4c5269088e914692e8f295c374e2)
- [MIT license at the tag](https://raw.githubusercontent.com/Farama-Foundation/Metaworld/3.0.0/LICENSE)
- [package metadata](https://github.com/Farama-Foundation/Metaworld/blob/3.0.0/pyproject.toml)

The official `pyproject.toml` at that tag declares package version `3.0.0`,
MIT, Python `>=3.10`, and dependencies including Gymnasium and MuJoCo.  The
separate `v3.0.0` tag points at an older commit whose package metadata still
says `2.0.0`; it is not the tag used by the upstream `metaworld==3.0.0`
installation route.

MuJoCo is an additional external dependency.  The upstream MetaWorld tag only
requires `mujoco>=3.0.0`, so this release does not invent a MuJoCo patch-level
pin.  Use a separately recorded MuJoCo version and license when qualifying a
real run; the official project is [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
under Apache-2.0.

The official RLinf model names provide public routes for the MetaWorld SFT and
FlowSDE model directories.  Immutable revisions and model-file digests were
recovered from official Hugging Face download-cache metadata, and the
manifest records the observed normalizer path
`lerobot/metaworld_mt50/norm_stats.json`.  Live Hub/API and card
revalidation, licenses, paper checkpoints, and a fixed evaluation panel are
still unresolved, so the caller must supply and verify them.  The repository's
evaluator is explicitly labelled a reconstructed evaluator and a custom panel
until a public formal panel artifact is available.

The official MetaWorld MT50 protocol evaluates 50 training goal states for
each of 50 tasks (2,500 episodes).  This release's 500-row (`50 x 10`) panel
is a reconstructed paper protocol, not that official 2,500-episode panel and
has no public content-addressed artifact.  The evaluator consequently labels
it `reconstructed_evaluator` and `custom`.

## What is still required before a paper-scale claim

The following items are intentionally not guessed or bundled:

1. CALVIN scene-D simulator files, task-oracle/annotation files, Official-D
   panel, and final training checkpoints, each with a public route, immutable
   revision, digest, and license.
2. The CALVIN FlowSDE teacher's license/redistribution terms.
3. ManiSkill package assets and any task-specific upstream sources beyond the
   RLinf dataset, including per-object Objaverse terms.
4. ManiSkill and MetaWorld base/teacher model directories, normalizers,
   checkpoints, and paper evaluation panels; model URLs alone are not enough
   without immutable revisions, file hashes, and license evidence.
5. A reproducible mapping from the `rlinf-openpi` wheel to a source commit, if
   source-level auditing (rather than wheel-level pinning) is required.

Until these are supplied and independently checked, synthetic smoke tests prove
the algorithmic and adapter contracts only; they do not prove that a real
simulator or the reported paper numbers have been reproduced.

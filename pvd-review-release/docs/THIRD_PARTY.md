# Third-Party Material Policy

Anonymous review does not waive attribution or license obligations. Upstream
creator names belong in required license and notice files even though project
author identities do not belong in the review artifact.

## Current artifact

The compact Python implementation is release-specific source. It imports
PyTorch and provides a compatibility adapter for a separately supplied
RLinf/OpenPI-style model. The repository includes a small source patch against
a pinned RLinf revision, together with the upstream Apache-2.0 license under
`third_party/rlinf/`. It does not vendor the RLinf tree, benchmark simulators,
model weights, datasets, or environment assets.

Runtime and test dependencies are obtained from their normal package indexes:

| Dependency | Role | Handling |
|---|---|---|
| Python | interpreter | external runtime; retain upstream terms |
| PyTorch | tensor and optimization runtime | external package; retain upstream terms |
| pytest | test-only | external package; retain upstream terms |
| Ruff | development-only static checker | external package; retain upstream terms |

This table is not a software bill of materials and does not substitute for the
license metadata shipped by those distributions.

The benchmark manifests now record limited public metadata. The ManiSkill
simulator asset repository card reports `ODC-By`, the CALVIN SFT repository
reports `Apache-2.0`, and public model routes are recorded for the ManiSkill and
MetaWorld SFT/FlowSDE names. For those ManiSkill and MetaWorld model routes,
immutable revisions and model-file hashes are recorded from official download
cache metadata; live Hub/API and license confirmation, complete directory
closure, and redistribution status remain pending. CALVIN environment assets,
panels, annotations, checkpoints, and the FlowSDE teacher license also remain
unresolved. The ManiSkill card notes that Objaverse-derived objects may have
additional object-level terms. A URL, repository card, or digest is not proof
that every file may be redistributed. The selected route/revision records were
cross-checked through a compatible API mirror when the canonical endpoint was
temporarily unavailable; this metadata check is not a download or
redistribution claim. The project-level license remains intentionally
undecided below.

## Rules for later additions

Before adding third-party code, a patch, model, dataset, panel, simulator asset,
or binary:

1. Record its canonical project/source, exact revision, retrieved filename,
   SHA-256, license identifier, and redistribution status.
2. Preserve the complete upstream license, notice, copyright, patent notice,
   citation request, and modification notice required by that version.
3. Put vendored notices under `third_party/<component>/`; do not merge upstream
   authors into project package metadata.
4. State whether a file is copied, generated, transformed, or patched, and keep
   the corresponding source or patch when the license requires it.
5. Do not mirror gated or non-redistributable material. Point to the canonical
   third-party process without adding author-controlled access approval.
6. Re-run the anonymity scanner and a license/SBOM review on the exact archive.

The anonymity scanner permits contact addresses only in attribution files under
`third_party/`. That exception exists to preserve required upstream text; it is
not permission to put project contacts there.

## Project license

This review snapshot intentionally has no project-level license because rights
holder and institutional approval have not been recorded in the release
evidence. Before describing the repository as open source, add an approved
OSI-compatible license or other intended grant, and verify that every included
dependency and asset is compatible with it. Do not use the project license to
override narrower third-party terms.

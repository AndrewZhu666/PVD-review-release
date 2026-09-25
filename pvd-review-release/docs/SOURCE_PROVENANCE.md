# Scientific-source provenance

This release is a benchmark-independent extraction of the scientific logic used
for the reported Path-OPD experiments. It does not vendor experiment
orchestration, cluster recovery code, tracking integrations, or private asset
locations.

## Authoritative source snapshot

The extraction was checked against the following archived source objects. The
identifiers are SHA-256 digests of the exact source bytes; paths are
repository-relative names from the research implementation.

| Archived source | SHA-256 | Released responsibility |
|---|---|---|
| `RLinf algorithm source (archived digest)` | `49d324d46e02e5fbea19fc10054371bb2efac211cbab8e241b3e5bbd7abcf883` | weighted path loss and frozen-teacher contract |
| `rlinf/models/embodiment/action_contract.py` | `de0d390820bada696ee6b43212d4826a5c1c6e162c412ee9325ff4135d6006f0` | executable action-domain crop |
| `RLinf K8 protocol helper (archived digest)` | `868135a9c1915da5d9e88098dd1f7036e9261136302dab18aba2de8f1e1e0376` | normalized flow grid and endpoint-exclusion shape contract |

These hashes are provenance identifiers, not runtime dependencies. Reproduction
from this repository does not require access to the archived tree.

## Semantic mapping

| Released API | Preserved scientific behavior |
|---|---|
| `ActionContract.crop` | keep only the executed action prefix and physical action dimensions |
| `FlowSchedule.uniform` | use pre-transition times `1 - i/K` and uniform weights `1/K` for `i = 0, ..., K-1` |
| `RolloutTrace.from_chains` | select `chains[:, :-1]`, excluding the endpoint, then detach and clone the sampled student path |
| `PathOPD.loss_from_velocities` | crop both velocity fields, detach the teacher target, average squared error over batch, horizon, and action dimensions at each time, then apply time weights |
| `FrozenTeacher` | force evaluation mode, disable parameter gradients, execute velocity queries under `no_grad`, and detect parameter mutation |

`tests/test_core_parity.py` anchors this mapping with independent, fixed
vectors. In particular, its two-time example has literal per-time losses
`[7.5, 3.0]` and weighted loss `4.125`; deliberately large differences outside
the executable domain must have no effect. The expected gradient is also stored
as a literal tensor, which verifies both the reduction axes and teacher
detachment.

`tests/test_core_invariants.py` observes the public supervision interface to
verify that the endpoint never reaches a velocity query, rollout states cannot
receive gradients, and the teacher remains frozen and unchanged.

## Integration and resume evidence

`tests/test_openpi_adapter.py` exercises the public OpenPI adapter against a
recording host-model boundary. It verifies the archived context expansion
order, detached context tensors, actor-only NFT query, and the detached,
flattened endpoint target passed through the model's native conditional
flow-matching preparation method.

`tests/test_toy_smoke.py` runs the real core, optimizer, RNG generator, and
checkpoint serializer on CPU. A run interrupted halfway must finish with the
same student parameter SHA-256 as an uninterrupted run. The test also repeats
the proof in a fresh workspace and requires identical results.

## Scope of the parity claim

These tests establish parity for the shared mathematical core and its gradient
boundaries. They do not claim byte-for-byte equivalence of simulator adapters,
model checkpoints, or benchmark environments. Benchmark reproduction requires
the separately documented versions, assets, and evaluation protocols.

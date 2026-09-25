# Runner synthetic smoke

Each benchmark training entrypoint has an explicit `--synthetic-smoke` mode:

```bash
python benchmarks/maniskill/train.py --synthetic-smoke \
  --work-dir artifacts/runner-smoke/maniskill \
  --output artifacts/runner-smoke/maniskill/report.json

python benchmarks/calvin_abc_d/train.py --synthetic-smoke \
  --work-dir artifacts/runner-smoke/calvin_abc_d \
  --output artifacts/runner-smoke/calvin_abc_d/report.json

python benchmarks/metaworld_mt50/train.py --synthetic-smoke \
  --work-dir artifacts/runner-smoke/metaworld_mt50 \
  --output artifacts/runner-smoke/metaworld_mt50/report.json
```

The mode is a runner-level contract check. It enters the selected benchmark
worker through its public command, then uses the release's deterministic
synthetic OpenPI-compatible policy and in-process environment. The report
schema is `path-opd-runner-synthetic-smoke-v1` and must contain:

```text
synthetic_only: true
runner_entrypoint_executed: true
external_runtime_executed: false
external_assets_used: false
real_simulator: false
gpu_executed: false
paper_scale: false
```

The nested benchmark report checks both Path-OPD and Endpoint-DAgger, the
configured solver/action dimensions, physical action prefix, frozen teacher,
optimizer update, checkpoint reload, exact model/optimizer/generator resume,
and the action values recorded at the synthetic environment boundary. A PASS
therefore proves that the released algorithm contract is reachable through
each runner; it is not a ManiSkill, CALVIN, or MetaWorld score and does not
replace a real run with caller-supplied assets and CUDA.

The ordinary runner commands retain their required external arguments and
fail closed when a real RLinf checkout, model, checkpoint, normalizer,
simulator, or panel is missing. `--synthetic-smoke` is an explicit alternate
mode and cannot be combined with real-run options.

Each evaluator also exposes the same dependency-light boundary proof:

```bash
python benchmarks/maniskill/evaluate.py --synthetic-smoke
python benchmarks/calvin_abc_d/evaluate.py --synthetic-smoke
python benchmarks/metaworld_mt50/evaluate.py --synthetic-smoke
```

Evaluator reports use schema `path-opd-evaluator-synthetic-smoke-v1` and
contain `claim: "synthetic_evaluator_contract_smoke_only"`. They first run
the deterministic training smoke to create and reload a synthetic checkpoint,
then validate a fixed-denominator synthetic panel digest and the exact
model-action-to-environment action trace. The evaluator creates one synthetic
rollout and one environment episode per ordered panel row, and reports the
ordered identity digest again from the traces; `panel_action_rows` and
`panel_identity_binding` therefore fail if the panel is not actually
consumed. They explicitly report
`external_runtime_executed: false`, `external_assets_used: false`,
`real_simulator: false`, `gpu_executed: false`, and `paper_scale: false`.
This is train/checkpoint/evaluator contract evidence only; it is not a
benchmark score or a substitute for a real simulator evaluation.

To execute all six public commands as subprocesses and validate their reports
in one step, use:

```bash
python scripts/benchmark_end_to_end_smoke.py \
  --work-dir artifacts/three-benchmark-e2e-smoke \
  --output artifacts/three-benchmark-e2e-smoke/report.json
```

The resulting schema is `path-opd-three-benchmark-e2e-smoke-v1` and the claim
is `synthetic_three_benchmark_end_to_end_only`. It remains explicitly
asset-free and synthetic-only.

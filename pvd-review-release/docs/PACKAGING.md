# Packaging and Installation Metadata

`pyproject.toml` defines a small `src`-layout package, the `path-opd` console
script, a Python 3.10 minimum, PyTorch as the runtime dependency, and pytest/Ruff
as optional test tools. Author, maintainer, contact, project URL, and citation
fields are intentionally absent during anonymous review.

## Supported review installation

The source checkout is the canonical artifact:

```bash
python -m pip install -e '.[test]'
path-opd smoke --work-dir artifacts/toy-smoke
```

A regular library install is also supported:

```bash
python -m pip install .
path-opd smoke --work-dir artifacts/toy-smoke
```

The installed wheel contains the Python package and CLI. Top-level configs,
historical result JSON, documentation, Docker files, and tests remain source
artifact data; they are not currently package data. Do not distribute only a
wheel when claiming to distribute the review artifact.

### Current wheel check

On 2026-09-22, `pip wheel --no-build-isolation --no-deps .` completed for the
release tree. The resulting wheel was installed with `--no-deps` into a fresh
Python 3.12 virtual environment, while using the already validated CPU PyTorch
environment for the runtime dependency. The installed `path-opd smoke` command
returned `status=PASS` and `interrupted.exact_resume=true`. This validates wheel
contents and the console entry point; it does not close the dependency, Docker,
GPU, or benchmark-asset gates.

## Before a public package release

Complete these metadata decisions after the anonymity period or with approved
anonymous wording:

- select and add the project license; add the corresponding SPDX expression to
  package metadata;
- add verified project URLs only after they no longer compromise review;
- add authors, maintainers, citation metadata, and acknowledgements only when
  double-blind restrictions no longer apply;
- decide whether configs and result schemas need a packaged-resource API, then
  include and test them in both wheel and source archive;
- replace broad dependency ranges with a tested constraints or lock file for
  the reproduction environment;
- verify the recorded container base digest on a registry-capable builder and
  add hashes for downloaded wheels;
- add classifiers and supported-platform statements only for platforms that
  have actually passed tests;
- build wheel and source archive in a clean environment, inspect their member
  lists and metadata, install each into a new environment, and rerun smoke;
- produce an SBOM and retain all third-party notices described in
  [THIRD_PARTY.md](THIRD_PARTY.md).

Do not publish the review placeholder to a public package index merely to test
installation: an account namespace or upload history can reveal provenance.

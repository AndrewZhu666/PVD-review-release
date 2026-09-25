FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b

ARG TORCH_VERSION=2.11.0
ARG PYTEST_VERSION=9.1.1
ARG SETUPTOOLS_VERSION=80.9.0
ARG WHEEL_VERSION=0.45.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/path-opd

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY benchmarks ./benchmarks
COPY docs ./docs
COPY results ./results
COPY integrations ./integrations
COPY third_party ./third_party
COPY scripts ./scripts
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/path-opd-entrypoint

RUN python -m pip install --no-cache-dir \
        "setuptools==${SETUPTOOLS_VERSION}" \
        "wheel==${WHEEL_VERSION}" \
        "pytest==${PYTEST_VERSION}" \
    && python -m pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
    && python -m pip install --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        --editable . \
    && chmod 0755 /usr/local/bin/path-opd-entrypoint \
    && groupadd --gid 10001 reviewer \
    && useradd --uid 10001 --gid 10001 --create-home reviewer \
    && mkdir -p /workspace/artifacts \
    && chown -R 10001:10001 /workspace

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/path-opd-entrypoint"]
CMD ["smoke"]

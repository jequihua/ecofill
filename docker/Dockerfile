# Build from the public repository root. CPython 3.12, Linux amd64, CPU wheels only.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS wheel
WORKDIR /build
COPY docker/requirements-linux-amd64.lock /build/requirements.lock
COPY --from=parquet_wheel /pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl /arrow/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
RUN python -c "import hashlib,pathlib; p=next(pathlib.Path('/arrow').glob('*.whl')); assert hashlib.sha256(p.read_bytes()).hexdigest() == '5389cdf79447ed1515c9e31620e6e1e2302249564d603f2ad727d4f6d313e4c3'" \
    && python -m pip install --no-index --no-deps /arrow/*.whl
RUN python -m pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /build/
COPY src /build/src
RUN python -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TABPFN_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    SKB_DATA_DIRECTORY=/tmp/skrub_data MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ecofill && useradd --uid 10001 --gid 10001 --no-create-home ecofill \
    && mkdir -p /work /opt/ecofill && chown 10001:10001 /work
COPY docker/requirements-linux-amd64.lock /opt/ecofill/requirements.lock
# Both stages share the exact locked environment; Arrow is acquired once through
# the explicit named build context, never downloaded a second time by pip.
COPY --from=wheel /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=wheel /usr/local/bin /usr/local/bin
COPY --from=wheel /wheels /opt/ecofill/wheels
RUN python -m pip install --no-deps /opt/ecofill/wheels/*.whl \
    && python -m pip uninstall -y wheel && python -m pip check \
    && python -m pip list --format=json > /opt/ecofill/python-packages.json \
    && dpkg-query -W > /opt/ecofill/system-packages.txt
COPY LICENSE THIRD_PARTY_NOTICES.md /opt/ecofill/
COPY docker/dependency-inventory.json /opt/ecofill/dependency-inventory.json
ENV ECOFILL_WHEEL_PATH=/opt/ecofill/wheels/ecofill-0.0.1-py3-none-any.whl
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["ecofill"]

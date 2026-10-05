FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ARG CLIPPING_TOOL_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOST=0.0.0.0 \
    PORT=8000 \
    CLIPPING_TOOL_DATA_ROOT=/data/clipping-tool \
    CLIPPING_TOOL_VERSION=${CLIPPING_TOOL_VERSION} \
    COOKIE_SECURE=0 \
    MAX_CONCURRENT_JOBS=1 \
    MAX_SOURCE_HEIGHT=1080 \
    RETENTION_DAYS=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        fontconfig \
        fonts-liberation2 \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/clipping-tool

COPY requirements.lock.txt ./
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --requirement requirements.lock.txt

COPY app ./app
COPY docker/entrypoint.sh /usr/local/bin/clipping-tool-entrypoint
RUN chmod 0755 /usr/local/bin/clipping-tool-entrypoint

EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl --fail --silent --show-error --max-time 5 \
        http://127.0.0.1:8000/login > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/clipping-tool-entrypoint"]

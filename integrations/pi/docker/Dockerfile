FROM node:22.19.0-bookworm-slim AS pi-build

WORKDIR /build/integrations/pi

# Install from the lock file so the image always contains the repository's
# pinned Pi 0.84.1 dependency graph.
COPY integrations/pi/package.json integrations/pi/package-lock.json ./
RUN npm ci

COPY integrations/pi/tsconfig.json ./
COPY integrations/pi/src ./src
COPY integrations/pi/assets ./assets

RUN npm run build \
    && npm prune --omit=dev


FROM node:22.19.0-bookworm-slim AS node-runtime


FROM python:3.12-slim@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    PI_CODING_AGENT_DIR=/pi-agent

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libglx-mesa0 \
        libsm6 \
        libstdc++6 \
        libxext6 \
        libxrender1 \
        ripgrep \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# The official Node image installs the statically bundled runtime here. npm is
# only needed in the build stage; the persistent Agent process only needs node.
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node

WORKDIR /opt/cadgenbench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Pin the authoring kernel used by generated programs. The orchestrator records
# the final image ID, which fixes the fully resolved Python environment.
RUN python -m pip install --no-cache-dir "build123d==0.10.0" .

WORKDIR /opt/cadgenbench-pi
COPY --from=pi-build /build/integrations/pi/package.json ./package.json
COPY --from=pi-build /build/integrations/pi/package-lock.json ./package-lock.json
COPY --from=pi-build /build/integrations/pi/node_modules ./node_modules
COPY --from=pi-build /build/integrations/pi/dist ./dist
COPY --from=pi-build /build/integrations/pi/assets ./assets

COPY integrations/pi/docker/entrypoint.sh /usr/local/bin/cadgenbench-pi-entrypoint

RUN useradd --create-home --home-dir /home/cadgen --uid 10001 cadgen \
    && mkdir -p /workspace /pi-agent /tmp/home /tmp/cache \
    && chown -R cadgen:cadgen /workspace /pi-agent /tmp/home /tmp/cache \
    && chmod 0555 /usr/local/bin/cadgenbench-pi-entrypoint

ENV CADGENBENCH_REPOSITORY_ROOT=/opt/cadgenbench \
    PATH=/opt/cadgenbench-pi/node_modules/.bin:${PATH}

# Runtime may replace this identity with the host uid/gid via docker --user so
# bind-mounted workspaces and Pi credential state remain writable.
USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/cadgenbench-pi-entrypoint"]
CMD ["node", "/opt/cadgenbench-pi/dist/src/container-agent.js"]

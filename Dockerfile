# syntax=docker/dockerfile:1
#
# Two things are built here: the Python service, and the frontend it serves.
#
# The Python half is pinned to linux/amd64 because mediapipe==0.10.21 publishes no Linux
# wheel for aarch64 (only manylinux_2_28_x86_64), so it must run under emulation on Apple
# Silicon. Docker Desktop provides that automatically; no extra flags at build or run time.
#
# The frontend half is not: its output is JavaScript, which is the same whatever built it, so
# it builds natively on the host's own architecture. Under emulation this stage would be the
# slowest part of the image for no benefit at all.

# --- the frontend ------------------------------------------------------------------------
FROM --platform=$BUILDPLATFORM node:22-slim AS web-build

WORKDIR /web

# Dependencies before sources, so editing a component does not reinstall node_modules.
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# --- the Python environment --------------------------------------------------------------
# Everything the tests and the dev shell need. `docker compose build dev` stops here, so a
# dev image never waits on the frontend build.
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# Runtime shared libraries mediapipe and opencv-python (non-headless) need that the
# slim base image doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies before the rest of the source so edits to application code
# don't invalidate this layer — only a pyproject.toml change does.
COPY pyproject.toml README.md LICENSE ./
COPY aced/__init__.py aced/__init__.py
RUN pip install --upgrade pip && pip install -e ".[api,dev]"

COPY . .

CMD ["bash"]

# --- the service -------------------------------------------------------------------------
# The runtime plus the built frontend, so one container serves the API and the web app from
# one origin. `web/dist` is gitignored and .dockerignored, so this copy is the only way it
# gets in — the image never carries a stale build from someone's working copy.
FROM runtime AS service

COPY --from=web-build /web/dist /app/web/dist

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

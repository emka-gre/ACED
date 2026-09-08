# syntax=docker/dockerfile:1
#
# Dev/CI image providing the pinned Python 3.11 + mediapipe + tensorflow environment
# (see pyproject.toml) on machines that can't install it natively — no local Python
# 3.11, or an Apple Silicon Mac where tensorflow==2.16.2 has no macOS arm64 wheel.
#
# Pinned to linux/amd64: mediapipe==0.10.21 publishes no Linux wheel for aarch64
# (only manylinux_2_28_x86_64), so this must run under emulation on Apple Silicon.
# Docker Desktop provides that automatically; no extra flags needed at build/run time.
FROM --platform=linux/amd64 python:3.11-slim

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

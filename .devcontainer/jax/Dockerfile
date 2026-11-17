FROM aerial-svd-jax:cuda12

# Editor/runtime tools only; retain the backend's pinned numerical dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    debugpy==1.8.20 ipykernel==6.29.5 ruff==0.11.13 && \
    python -m pip check

# Dev Containers updates these IDs to the Linux host.
RUN groupadd --gid 1000 research && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash research
USER research

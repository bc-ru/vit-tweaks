FROM ghcr.io/bc-ru/pytorch-2.1.0-cuda11.8-cudnn8-runtime-timm-models:stable
RUN groupadd --gid 1000 research && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash research
USER research

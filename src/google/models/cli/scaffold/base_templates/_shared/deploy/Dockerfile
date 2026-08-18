# Copyright 2026 Google LLC
#
# Custom Serving Container Recipe for Vertex AI / GEAP

ARG BASE_IMAGE=us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Upgrade transformers / tokenizers for latest model support
RUN pip install --no-cache-dir --upgrade transformers huggingface-hub

WORKDIR /workspace

# Copy custom entrypoint
COPY ./entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]

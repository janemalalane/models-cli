# models-cli (`google-models-cli`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-enabled-brightgreen.svg)](https://github.com/astral-sh/uv)

A developer-first CLI toolchain to evaluate, optimize, deploy, chat with, and benchmark open-weights GenAI models (Gemma, Llama, Qwen, DeepSeek, etc.) on **Google Cloud (Gemini Enterprise)** using **vLLM** and **SGLang**.

---

## Installation

### Ephemeral execution (no install needed)
Run directly via `uvx` pointing to the repository:
```bash
uvx --from "git+https://github.com/janemalalane/models-cli.git" models-cli --help
```

### Install as a global tool
Install permanently in your shell:
```bash
uv tool install "git+https://github.com/janemalalane/models-cli.git"
models-cli --help
```

### Local development
```bash
git clone https://github.com/janemalalane/models-cli.git
cd models-cli
uv sync --all-extras
uv run models-cli --help
```

> **Aliases:** Both `models-cli` and `models` are available.

---

## Core Commands Overview

| Command | Purpose | Key Flags |
|---|---|---|
| [`info`](#diagnostics-models-cli-info) | Check active GCP project, credentials, and configuration | — |
| [`create`](#1-scaffold-a-serving-project-models-cli-create) | Scaffold a new model-serving project with configs & recipes | `-i/--interactive`, `-t/--template`, `-m/--model`, `-y/--yes` |
| [`eval`](#2-evaluate-candidate-models-models-cli-eval) | Evaluate model quality against golden datasets & pick a winner | `-m/--models`, `-d/--dataset`, `--set-winner`, `--mock` |
| [`recommend`](#3-hardware-recommendations-models-cli-recommend) | Zero-shot GPU/TPU & engine tuning recommendations via GKE Recommender | `-m/--model`, `-u/--use-case`, `-f/--family`, `-s/--sort-by` |
| [`deploy`](#4-deploy-to-gemini-enterprise-online-prediction-models-cli-deploy) | Deploy to Gemini Enterprise Online Prediction endpoint, check status, or watch | `--dry-run`, `--status`, `-w/--watch`, `-b/--bucket` |
| [`playground`](#5-test-endpoints-interactively-models-cli-playground) | Interactive streaming chat or prompt testing with deployed models | `-m/--message`, `-s/--system`, `-e/--endpoint`, `--raw` |
| [`benchmark`](#6-benchmark-performance-models-cli-benchmark) | Measure TTFT, TPOT, & throughput using `inference-perf` | `-e/--endpoint`, `-c/--config`, `--mock` |

---

## Workflow Guide

### Diagnostics: `models-cli info`
Inspect resolved GCP project, credentials, active region, and model settings:
```bash
models-cli info
```

---

### 1. Scaffold a Serving Project: `models-cli create`
Generate a workspace with serving configs (`engine_config.yaml`, `deployment_spec.yaml`), golden evaluation sets, and benchmark configs:
```bash
# Interactive setup (prompts for model, engine, GCP region, bucket, etc.)
models-cli create -i

# Quickstart non-interactive (defaults to vLLM on Google Gemma)
models-cli create bedtime-story-writer --model google/gemma-4-31B-it -y
cd bedtime-story-writer
```

---

### 2. Evaluate Candidate Models: `models-cli eval`
Score multiple open-weights models on task-specific test cases before selecting hardware:
```bash
# Compare candidate models on golden dataset & persist the winner to project configs
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it" --set-winner

# Offline mock evaluation (useful for CI/CD checks)
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it" --mock
```

---

### 3. Hardware Recommendations: `models-cli recommend`
Query the Google Cloud GKE Recommender for optimized accelerator configurations (TPU v5e/v6e, NVIDIA L4/A100) and engine parameters:
```bash
# Find lowest hourly cost setup for Gemma
models-cli recommend --model google/gemma-4-31B-it --sort-by cost

# Target low Time to First Token (TTFT) on Google TPUs
models-cli recommend --model google/gemma-4-31B-it --family tpu --sort-by ttft

# Tailor for specific traffic patterns (e.g. summarization, chatbot, code-completion)
models-cli recommend --model google/gemma-4-31B-it --use-case summarization

# Apply recommendations interactively directly to deployment_spec.yaml
models-cli recommend --model google/gemma-4-31B-it --apply
```

---

### 4. Deploy to Gemini Enterprise Online Prediction: `models-cli deploy`
Register model artifacts and deploy container instances to a Gemini Enterprise Online Prediction dedicated endpoint:
```bash
# 1. Validate configuration and manifest without cloud calls
models-cli deploy --dry-run

# 2. Deploy to Gemini Enterprise Online Prediction (asynchronous by default, tracks operation in metadata)
models-cli deploy

# 3. Check status of current or specific long-running deployment
models-cli deploy --status

# 4. Stream status and live logs until endpoint deployment is active
models-cli deploy --status --watch
```

---

### 5. Test Endpoints Interactively: `models-cli playground`
Chat directly with your deployed endpoint using the OpenAI-compatible Chat Completions API with streaming and reasoning token support:
```bash
# Launch interactive multi-turn chat session (auto-discovers deployed endpoint)
models-cli playground

# Send a single quick query
models-cli playground "Tell me a bedtime story about a curious robotic fox."

# Custom system prompt with explicit endpoint
models-cli playground -m "Summarize this log" -s "You are an SRE assistant." -e <ENDPOINT_ID>
```

---

### 6. Benchmark Performance: `models-cli benchmark`
Execute production-grade load tests with `inference-perf` to profile TTFT, TPOT, and concurrency curves:
```bash

# Run live benchmark against your active deployed endpoint
models-cli benchmark

# Benchmark with a custom endpoint and load profile
models-cli benchmark --endpoint "https://<ENDPOINT_DNS>/v1" --config tests/benchmark/config.yaml
```
Markdown executive summaries and metric distributions are automatically written to `reports/`.

---

## Workspace Structure

When you scaffold a project (`models-cli create`), the resulting directory structure is:

```text
my-service/
├── README.md                      # Service setup and quickstart
├── pyproject.toml                 # uv project dependencies
├── .env.example                   # GCP and Hugging Face environment variables
│
├── config/
│   ├── engine_config.yaml         # vLLM / SGLang serving flags (e.g. max_model_len, tensor_parallel)
│   └── deployment_spec.yaml       # Hardware (machine type, accelerators) & container spec
│
├── tests/
│   ├── eval/
│   │   └── golden_dataset.jsonl   # Task-specific prompt / ground-truth pairs
│   └── benchmark/
│       ├── config.yaml            # inference-perf traffic distribution & concurrency targets
│       └── prompts.jsonl          # Benchmark test dataset
│
└── reports/                       # Generated benchmark analyses, eval rankings, and charts
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

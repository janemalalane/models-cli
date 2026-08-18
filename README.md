# models-cli (`google-models-cli`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-enabled-brightgreen.svg)](https://github.com/astral-sh/uv)

A formal command-line interface toolchain to evaluate, optimize, deploy, and benchmark open-weights GenAI models (Gemma, Llama, GLM, etc.) on **Google Cloud (Gemini Enterprise Agent Platform / Vertex AI)**.

Structured following the architectural design of [`google/agents-cli`](https://github.com/google/agents-cli).

---

## Key Features

- 🏗️ **`models-cli create`**: Interactive and automated project scaffolding for open model serving workspaces with unified evaluation, benchmarking, and engine configs.
- ⚡ **`models-cli recommend`**: Zero-shot hardware configuration and vLLM / SGLang parameter recommendations optimized for TTFT, TPOT, throughput, or hourly budget across Google Cloud GPUs (L4, G4, A100) and TPUs (v5e, Trillium v6e).
- 🚀 **`models-cli deploy`**: Seamless model registration and online prediction endpoint provisioning on Vertex AI / GEAP with pre-built Google vLLM / SGLang containers or custom images, with `--dry-run` validation.
- 📊 **`models-cli benchmark`**: Automated performance load testing using `inference-perf`, parsing per-request lifecycle metrics (TTFT, TPOT, goodput), and generating executive Markdown reports.
- 🔍 **`models-cli eval`**: Multi-model response accuracy & quality evaluation against golden test datasets, ranking the winning model and syncing project configurations.

---

## Quickstart: End-to-End Walkthrough

Here is how to evaluate, configure, deploy, and benchmark a real-world open-weights service — **`bedtime-story-writer`** — using Google Cloud Vertex AI / Gemini Enterprise Agent Platform.

### 1. Installation

Run `models-cli` directly with **`uvx`**:
```bash
uvx google-models-cli --help
```

Or install in your local environment with **`uv`**:
```bash
uv sync --all-extras
uv run models-cli --help
```

### 2. Scaffold the Project

Create a new model serving and benchmarking workspace for `bedtime-story-writer`:
```bash
# Non-interactive quickstart (defaults to Google Gemma 4 on vLLM)
models-cli create bedtime-story-writer --model google/gemma-4-31B-it -y

# Move into the project directory
cd bedtime-story-writer
```

### 3. Evaluate Candidate Models

Evaluate candidate open-weights models against your bedtime story golden test dataset (`tests/eval/golden_dataset.jsonl`) to select the best performer:
```bash
# Evaluate models and automatically set the winner in project configs
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it" --set-winner

# Or run in mock mode for offline testing / CI
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it" --mock
```

### 4. Zero-Shot Hardware Recommendation

Get optimal accelerator and engine parameters tailored for low Time to First Token (TTFT) or minimum hourly cost:
```bash
# Optimize for lowest TTFT on Google Cloud TPUs
models-cli recommend --model google/gemma-4-31B-it --objective ttft --family tpu

# Or optimize for lowest cost across all accelerator types
models-cli recommend --model google/gemma-4-31B-it --objective cost
```

### 5. Deploy to Vertex AI / GEAP Online Endpoint

Validate your deployment manifest, then provision a dedicated online prediction endpoint on Google Cloud:
```bash
# Validate manifests without executing cloud calls
models-cli deploy --dry-run

# Deploy to live Vertex AI / GEAP endpoint
models-cli deploy
```

### 6. Benchmark Serving Performance

Execute load tests with `inference-perf` to measure TTFT, TPOT, and throughput curves, generating markdown reports in `reports/`:
```bash
# Run simulated benchmark against mock server
models-cli benchmark --mock

# Run live benchmark against deployed endpoint
models-cli benchmark
```

---

## CLI Command Reference

### 1. Project Scaffolding
Create a new self-contained model serving and benchmarking project:
```bash
# Interactive mode (prompts for project name, model, engine, region, and verifies GCP credentials)
models-cli create

# Quickstart non-interactive mode
models-cli create bedtime-story-writer --model google/gemma-4-31B-it -y
```

### 2. Zero-Shot Hardware Recommendation
Find the optimal accelerator and engine parameters for your model:
```bash
# Optimize for lowest hourly cost
models-cli recommend --model google/gemma-4-31B-it --objective cost

# Optimize for lowest Time to First Token (TTFT) on TPUs
models-cli recommend --model google/gemma-4-31B-it --objective ttft --family tpu

# Output structured JSON for automation
models-cli recommend --model google/gemma-4-31B-it --format json
```

### 3. Model Response Quality Evaluation
Evaluate candidate open models on a golden test dataset before deployment:
```bash
# Evaluate models and select the winner
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it"

# Mock evaluation for offline testing / CI
models-cli eval --models "google/gemma-4-31B-it,google/gemma-2-27b-it" --mock
```

### 4. Deploy to Vertex AI / GEAP Online Endpoint
Deploy model container to a dedicated GEAP prediction endpoint:
```bash
# Validate manifests without executing cloud calls
models-cli deploy --model-id gemma-31b --machine-type ct6e-standard-4t --dry-run

# Deploy to live Vertex AI endpoint
models-cli deploy --model-id gemma-31b --machine-type ct6e-standard-4t
```

### 5. Performance Benchmarking
Execute load tests and measure latency curves with `inference-perf`:
```bash
# Run simulated benchmark against mock server
models-cli benchmark --mock

# Run live benchmark against deployed endpoint
models-cli benchmark --endpoint "https://us-central1-aiplatform.googleapis.com/v1/..."
```

---

## Project Structure

```text
bedtime-story-writer/
├── README.md                      # Quickstart and command instructions
├── pyproject.toml                 # uv project dependencies
├── .env.example                   # GCP and HuggingFace environment variables
│
├── config/                        # Model & engine runtime configurations
│   ├── engine_config.yaml         # Engine parameters (vLLM / SGLang settings)
│   └── deployment_spec.yaml       # Hardware & container deployment specification
│
├── deploy/                        # Container recipes (for custom image builds)
│   ├── Dockerfile                 # Custom container build recipe
│   └── entrypoint.sh              # Container entrypoint script
│
├── tests/                         # Unified evaluation & benchmark test suite
│   ├── eval/                      # Response quality & accuracy evaluation
│   │   └── golden_dataset.jsonl   # Test dataset with expected inputs/outputs
│   └── benchmark/                 # Performance & load testing
│       ├── config.yaml            # inference-perf traffic/protocol configuration
│       └── prompts.jsonl          # Load test prompt distribution
│
└── reports/                       # Generated benchmark results, plots, and eval reports
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

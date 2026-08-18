# {project_name}

Open-weights GenAI Model Serving, Deployment & Benchmarking workspace for **{model_id}** on **Gemini Enterprise Agent Platform (GEAP) / Vertex AI**.

---

## Quickstart Lifecycle

### 1. Setup Environment
Ensure your `.env` file is configured with your Google Cloud credentials:
```bash
cp .env.example .env
```

### 2. Evaluate Candidate Models (Optional)
Run response quality evaluation across candidate open models on the golden dataset:
```bash
models-cli eval --models "{model_id},google/gemma-2-27b-it"
```

### 3. Recommend Hardware & Engine Parameters
Get zero-shot hardware recommendations optimized for TTFT, TPOT, throughput, or cost:
```bash
models-cli recommend --model {model_id} --objective ttft
```

### 4. Deploy to GEAP / Vertex AI Online Endpoint
Deploy the model container to a dedicated GEAP prediction endpoint:
```bash
# Dry run to verify deployment manifests
models-cli deploy --dry-run

# Execute deployment to Vertex AI
models-cli deploy
```

### 5. Benchmark Endpoint Performance
Run load testing and performance benchmarks using `inference-perf`:
```bash
# Run simulated / mock benchmark
models-cli benchmark --mock

# Run live benchmark against deployed endpoint
models-cli benchmark
```
Output reports and latency curves will be stored in `reports/`.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants and default configuration values for models-cli."""

from enum import Enum

# Google Pre-built Serving Container Images for Gemini Enterprise Online Prediction
DEFAULT_VLLM_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/agent-platform-mg-public/containers/pytorch-vllm-serve"
)
DEFAULT_SGLANG_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/agent-platform-mg-public/containers/pytorch-sglang-serve"
)


class InferenceEngine(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"

    @property
    def image_uri(self) -> str:
        images = {
            InferenceEngine.VLLM: DEFAULT_VLLM_CONTAINER_IMAGE,
            InferenceEngine.SGLANG: DEFAULT_SGLANG_CONTAINER_IMAGE,
        }
        return images[self]

    @property
    def env_vars(self) -> dict[str, str]:
        envs = {
            InferenceEngine.VLLM: {
                "VLLM_LOGGING_LEVEL": "INFO",
                "HF_HOME": "/dev/shm/hf",
                "TMPDIR": "/dev/shm",
                "LOCAL_DOWNLOAD_DIR": "/dev/shm",
                "LOCAL_MODEL_DIR": "/dev/shm/model_dir",
                "CLOUDSDK_STORAGE_MAX_RETRIES": "3",
            },
            InferenceEngine.SGLANG: {
                "SGLANG_LOGGING_LEVEL": "INFO",
                "HF_HOME": "/dev/shm/hf",
                "TMPDIR": "/dev/shm",
                "LOCAL_DOWNLOAD_DIR": "/dev/shm",
                "LOCAL_MODEL_DIR": "/dev/shm/model_dir",
                "CLOUDSDK_STORAGE_MAX_RETRIES": "3",
            },
        }
        return envs[self]


class AcceleratorFamily(str, Enum):
    GPU = "gpu"
    TPU = "tpu"
    ANY = "any"


class AcceleratorType(str, Enum):
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    NVIDIA_H200_141GB = "NVIDIA_H200_141GB"
    NVIDIA_B200 = "NVIDIA_B200"
    NVIDIA_GB200 = "NVIDIA_GB200"
    NVIDIA_L4 = "NVIDIA_L4"
    NVIDIA_RTX_PRO_6000 = "NVIDIA_RTX_PRO_6000"
    TPU_V5_LITEPOD = "TPU_V5_LITEPOD"
    TPU_V6E = "TPU_V6E"


class OptimizationMetric(str, Enum):
    TTFT = "ttft"  # Time to First Token
    TPOT = "tpot"  # Time per Output Token
    THROUGHPUT = "throughput"  # Requests or Tokens per second
    COST = "cost"  # Price per hour / Price per token


DEFAULT_MODEL_REPO = "google/gemma-4-31B-it"
DEFAULT_REGION = "us-central1"
DEFAULT_ENGINE = "vllm"
DEFAULT_PRICING_MODEL = "on-demand"


# Standard Container Routes & Networking
DEFAULT_PREDICTION_ROUTE = "/*"
DEFAULT_HEALTH_ROUTE = "/health"
DEFAULT_CONTAINER_PORT = 8080
DEFAULT_CONTAINER_HOST = "0.0.0.0"
DEFAULT_HOST = DEFAULT_CONTAINER_HOST
DEFAULT_SHARED_MEMORY_MB = 128 * 1024  # 128 GB

# Deployment & Upload Timeouts (seconds)
DEFAULT_UPLOAD_REQUEST_TIMEOUT = 3600  # 1 hour
DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT = 3600  # 1 hour
DEFAULT_MODEL_UPLOAD_TIMEOUT = DEFAULT_UPLOAD_REQUEST_TIMEOUT
DEFAULT_DEPLOYMENT_TIMEOUT = DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT


# Default Engine Parameters
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_NUM_SEQS = 512

# Standard Hardware Catalog for Gemini Enterprise Online Prediction
HARDWARE_SPECS: dict[str, dict] = {
    # Nvidia GPUs - A2 Series (NVIDIA A100)
    "a2-highgpu-1g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB",
        "accelerator_type": AcceleratorType.NVIDIA_TESLA_A100.value,
        "accelerator_count": 1,
        "vram_gb": 40,
        "hourly_cost_usd": 3.67,
        "memory_bandwidth_gb_s": 1555,
        "description": "High-bandwidth A100 GPU for low latency inference",
    },
    "a2-highgpu-2g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB (2x)",
        "accelerator_type": AcceleratorType.NVIDIA_TESLA_A100.value,
        "accelerator_count": 2,
        "vram_gb": 80,
        "hourly_cost_usd": 7.34,
        "memory_bandwidth_gb_s": 3110,
        "description": "Dual A100 (40GB) GPUs for distributed inference",
    },
    "a2-highgpu-4g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_TESLA_A100.value,
        "accelerator_count": 4,
        "vram_gb": 160,
        "hourly_cost_usd": 14.68,
        "memory_bandwidth_gb_s": 6220,
        "description": "Quad A100 (40GB) GPUs for multi-GPU inference",
    },
    "a2-highgpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_TESLA_A100.value,
        "accelerator_count": 8,
        "vram_gb": 320,
        "hourly_cost_usd": 29.36,
        "memory_bandwidth_gb_s": 12440,
        "description": "8x A100 (40GB) GPUs for large model serving",
    },
    "a2-megagpu-16g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB (16x)",
        "accelerator_type": AcceleratorType.NVIDIA_TESLA_A100.value,
        "accelerator_count": 16,
        "vram_gb": 640,
        "hourly_cost_usd": 58.72,
        "memory_bandwidth_gb_s": 24880,
        "description": "16x A100 (40GB) GPUs for ultra-large model serving",
    },
    "a2-ultragpu-1g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-80GB",
        "accelerator_type": AcceleratorType.NVIDIA_A100_80GB.value,
        "accelerator_count": 1,
        "vram_gb": 80,
        "hourly_cost_usd": 4.50,
        "memory_bandwidth_gb_s": 2039,
        "description": "Large memory A100 GPU for 30B-70B models",
    },
    "a2-ultragpu-2g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-80GB (2x)",
        "accelerator_type": AcceleratorType.NVIDIA_A100_80GB.value,
        "accelerator_count": 2,
        "vram_gb": 160,
        "hourly_cost_usd": 9.00,
        "memory_bandwidth_gb_s": 4078,
        "description": "Dual A100 (80GB) GPUs for large models with low latency",
    },
    "a2-ultragpu-4g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-80GB (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_A100_80GB.value,
        "accelerator_count": 4,
        "vram_gb": 320,
        "hourly_cost_usd": 18.00,
        "memory_bandwidth_gb_s": 8156,
        "description": "Quad A100 (80GB) GPUs for 70B+ model inference",
    },
    "a2-ultragpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-80GB (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_A100_80GB.value,
        "accelerator_count": 8,
        "vram_gb": 640,
        "hourly_cost_usd": 36.00,
        "memory_bandwidth_gb_s": 16312,
        "description": "8x A100 (80GB) GPU cluster for high-throughput 70B+ model serving",
    },
    # Nvidia GPUs - A3 Series (NVIDIA H100 / H200)
    "a3-highgpu-1g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB",
        "accelerator_type": AcceleratorType.NVIDIA_H100_80GB.value,
        "accelerator_count": 1,
        "vram_gb": 80,
        "hourly_cost_usd": 3.75,
        "memory_bandwidth_gb_s": 3350,
        "description": "Single H100 SXM5 GPU for high-performance LLM serving",
    },
    "a3-highgpu-2g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (2x)",
        "accelerator_type": AcceleratorType.NVIDIA_H100_80GB.value,
        "accelerator_count": 2,
        "vram_gb": 160,
        "hourly_cost_usd": 7.50,
        "memory_bandwidth_gb_s": 6700,
        "description": "Dual H100 GPU configuration for low-latency medium models",
    },
    "a3-highgpu-4g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_H100_80GB.value,
        "accelerator_count": 4,
        "vram_gb": 320,
        "hourly_cost_usd": 15.00,
        "memory_bandwidth_gb_s": 13400,
        "description": "Quad H100 GPUs for multi-GPU inference up to 70B models",
    },
    "a3-highgpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_H100_80GB.value,
        "accelerator_count": 8,
        "vram_gb": 640,
        "hourly_cost_usd": 30.00,
        "memory_bandwidth_gb_s": 26800,
        "description": "Ultra-performance 8x H100 cluster for massive concurrency",
    },
    "a3-edgegpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_H100_80GB.value,
        "accelerator_count": 8,
        "vram_gb": 640,
        "hourly_cost_usd": 30.00,
        "memory_bandwidth_gb_s": 26800,
        "description": "Edge-optimized 8x H100 cluster for massive concurrency",
    },
    "a3-ultragpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H200-141GB (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_H200_141GB.value,
        "accelerator_count": 8,
        "vram_gb": 1128,
        "hourly_cost_usd": 35.00,
        "memory_bandwidth_gb_s": 38400,
        "description": "Ultra-performance 8x H200 (141GB) cluster for massive models and high concurrency",
    },
    # Nvidia GPUs - A4 Series (NVIDIA B200)
    "a4-highgpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia B200 (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_B200.value,
        "accelerator_count": 8,
        "vram_gb": 1440,
        "hourly_cost_usd": 40.00,
        "memory_bandwidth_gb_s": 64000,
        "description": "Next-gen 8x NVIDIA B200 Blackwell GPU cluster for frontier models",
    },
    # Nvidia GPUs - A4X Series (NVIDIA GB200)
    "a4x-highgpu-4g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia GB200 (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_GB200.value,
        "accelerator_count": 4,
        "vram_gb": 768,
        "hourly_cost_usd": 30.00,
        "memory_bandwidth_gb_s": 32000,
        "description": "Quad NVIDIA GB200 Grace Blackwell superchip node for ultra-high throughput",
    },
    # Nvidia GPUs - G2 Series (NVIDIA L4)
    "g2-standard-4": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 0.70,
        "memory_bandwidth_gb_s": 300,
        "description": "Entry-level single L4 GPU for small models (up to 7B FP8/INT4)",
    },
    "g2-standard-8": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 0.85,
        "memory_bandwidth_gb_s": 300,
        "description": "Cost-effective GPU for small models (up to 7B-9B FP8/INT4)",
    },
    "g2-standard-12": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 1.05,
        "memory_bandwidth_gb_s": 300,
        "description": "Single L4 GPU with 12 vCPUs for low-latency small models",
    },
    "g2-standard-16": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 1.25,
        "memory_bandwidth_gb_s": 300,
        "description": "Single L4 GPU with 16 vCPUs for medium memory CPU workloads",
    },
    "g2-standard-24": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4 (2x)",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 2,
        "vram_gb": 48,
        "hourly_cost_usd": 1.70,
        "memory_bandwidth_gb_s": 600,
        "description": "Dual L4 GPU setup for models up to 13B-15B FP8",
    },
    "g2-standard-32": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 1.75,
        "memory_bandwidth_gb_s": 300,
        "description": "Single L4 GPU with 32 vCPUs for CPU-intensive pre/post-processing",
    },
    "g2-standard-48": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4 (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 4,
        "vram_gb": 96,
        "hourly_cost_usd": 3.40,
        "memory_bandwidth_gb_s": 1200,
        "description": "Multi-GPU L4 setup for medium models (up to 30B FP8)",
    },
    "g2-standard-96": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4 (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_L4.value,
        "accelerator_count": 8,
        "vram_gb": 192,
        "hourly_cost_usd": 6.80,
        "memory_bandwidth_gb_s": 2400,
        "description": "8x L4 GPU configuration for models up to 70B FP8",
    },
    # Nvidia GPUs - G4 Series (NVIDIA RTX PRO 6000)
    "g4-standard-48": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000",
        "accelerator_type": AcceleratorType.NVIDIA_RTX_PRO_6000.value,
        "accelerator_count": 1,
        "vram_gb": 48,
        "hourly_cost_usd": 1.30,
        "memory_bandwidth_gb_s": 850,
        "description": "RTX PRO 6000 GPU for medium models (up to 30B FP8) and high-throughput inference",
    },
    "g4-standard-96": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000 (2x)",
        "accelerator_type": AcceleratorType.NVIDIA_RTX_PRO_6000.value,
        "accelerator_count": 2,
        "vram_gb": 96,
        "hourly_cost_usd": 2.60,
        "memory_bandwidth_gb_s": 1700,
        "description": "Nvidia RTX PRO 6000 Blackwell GPU for high-throughput LLM inference",
    },
    "g4-standard-192": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000 (4x)",
        "accelerator_type": AcceleratorType.NVIDIA_RTX_PRO_6000.value,
        "accelerator_count": 4,
        "vram_gb": 192,
        "hourly_cost_usd": 10.40,
        "memory_bandwidth_gb_s": 6800,
        "description": "Quad RTX PRO 6000 GPUs for multi-GPU inference and large context windows",
    },
    "g4-standard-384": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000 (8x)",
        "accelerator_type": AcceleratorType.NVIDIA_RTX_PRO_6000.value,
        "accelerator_count": 8,
        "vram_gb": 384,
        "hourly_cost_usd": 20.80,
        "memory_bandwidth_gb_s": 13600,
        "description": "8x RTX PRO 6000 GPU cluster for massive multi-GPU model serving",
    },
    # Google TPUs
    "ct5lp-hightpu-1t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google TPU v5e (1-chip)",
        "accelerator_type": AcceleratorType.TPU_V5_LITEPOD.value,
        "accelerator_count": 1,
        "vram_gb": 16,
        "hourly_cost_usd": 1.20,
        "memory_bandwidth_gb_s": 819,
        "description": "Cost-efficient TPU v5e single core",
    },
    "ct5lp-hightpu-4t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google TPU v5e (4-chip)",
        "accelerator_type": AcceleratorType.TPU_V5_LITEPOD.value,
        "accelerator_count": 4,
        "vram_gb": 64,
        "hourly_cost_usd": 4.80,
        "memory_bandwidth_gb_s": 3276,
        "description": "TPU v5e 4-chip pod for medium open models",
    },
    "ct6e-standard-1t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (1-chip)",
        "accelerator_type": AcceleratorType.TPU_V6E.value,
        "accelerator_count": 1,
        "vram_gb": 32,
        "hourly_cost_usd": 1.80,
        "memory_bandwidth_gb_s": 1640,
        "description": "Next-gen Trillium TPU optimized for low TPOT and high token throughput",
    },
    "ct6e-standard-4t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (4-chip)",
        "accelerator_type": AcceleratorType.TPU_V6E.value,
        "accelerator_count": 4,
        "vram_gb": 128,
        "hourly_cost_usd": 7.20,
        "memory_bandwidth_gb_s": 6560,
        "description": "Trillium TPU 4-chip configuration for high-concurrency production serving",
    },
    "ct6e-standard-8t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (8-chip)",
        "accelerator_type": AcceleratorType.TPU_V6E.value,
        "accelerator_count": 8,
        "vram_gb": 256,
        "hourly_cost_usd": 14.40,
        "memory_bandwidth_gb_s": 13120,
        "description": "Trillium TPU 8-chip configuration for ultra-scale LLM inference",
    },
}

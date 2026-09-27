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

# Default Engine Parameters
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_KV_CACHE_DTYPE = "fp8"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_MAX_NUM_SEQS = 512

# Standard Hardware Catalog for Gemini Enterprise Online Prediction
HARDWARE_SPECS: dict[str, dict] = {
    # Nvidia GPUs
    "a4-highgpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A4",
        "accelerator_type": "NVIDIA_B200",
        "accelerator_count": 8,
        "vram_gb": 3968,
        "hourly_cost_usd": 0.50,
        "memory_bandwidth_gb_s": 200,
        "description": "Cost-effective GPU for small models (up to 7B FP8/INT4)",
    },
    "g2-standard-8": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "vram_gb": 24,
        "hourly_cost_usd": 0.85,
        "memory_bandwidth_gb_s": 300,
        "description": "Cost-effective GPU for small models (up to 7B-9B FP8/INT4)",
    },
    "g2-standard-48": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia L4 (4x)",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 4,
        "vram_gb": 192,
        "hourly_cost_usd": 3.40,
        "memory_bandwidth_gb_s": 1200,
        "description": "Multi-GPU L4 setup for medium models (up to 30B FP8)",
    },
    "g4-standard-48": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000",
        "accelerator_type": "NVIDIA_RTX_PRO_6000",
        "accelerator_count": 1,
        "vram_gb": 48,
        "hourly_cost_usd": 1.30,
        "memory_bandwidth_gb_s": 850,
        "description": "RTX PRO 6000 GPU for medium models (up to 30B FP8) and high-throughput inference",
    },
    "g4-standard-96": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000",
        "accelerator_type": "NVIDIA_RTX_PRO_6000",
        "accelerator_count": 2,
        "vram_gb": 96,
        "hourly_cost_usd": 2.60,
        "memory_bandwidth_gb_s": 1700,
        "description": "Nvidia RTX PRO 6000 Blackwell GPU for high-throughput LLM inference",
    },
    "g4-standard-192": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000 (4x)",
        "accelerator_type": "NVIDIA_RTX_PRO_6000",
        "accelerator_count": 4,
        "vram_gb": 192,
        "hourly_cost_usd": 10.40,
        "memory_bandwidth_gb_s": 6800,
        "description": "Quad RTX PRO 6000 GPUs for multi-GPU inference and large context windows",
    },
    "g4-standard-384": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia RTX PRO 6000 (8x)",
        "accelerator_type": "NVIDIA_RTX_PRO_6000",
        "accelerator_count": 8,
        "vram_gb": 384,
        "hourly_cost_usd": 20.80,
        "memory_bandwidth_gb_s": 13600,
        "description": "8x RTX PRO 6000 GPU cluster for massive multi-GPU model serving",
    },
    "a2-highgpu-1g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-40GB",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "vram_gb": 40,
        "hourly_cost_usd": 3.67,
        "memory_bandwidth_gb_s": 1555,
        "description": "High-bandwidth A100 GPU for low latency inference",
    },
    "a2-ultragpu-1g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia A100-80GB",
        "accelerator_type": "NVIDIA_A100_80GB",
        "accelerator_count": 1,
        "vram_gb": 80,
        "hourly_cost_usd": 4.50,
        "memory_bandwidth_gb_s": 2039,
        "description": "Large memory A100 GPU for 30B-70B models",
    },
    "a3-highgpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (8x)",
        "accelerator_type": "NVIDIA_H100_80GB",
        "accelerator_count": 8,
        "vram_gb": 640,
        "hourly_cost_usd": 30.00,
        "memory_bandwidth_gb_s": 26800,
        "description": "Ultra-performance 8x H100 cluster for massive concurrency",
    },
    "a3-ultragpu-8g": {
        "family": AcceleratorFamily.GPU,
        "chip_name": "Nvidia H100-80GB (8x)",
        "accelerator_type": "NVIDIA_H200_141GB",
        "accelerator_count": 8,
        "vram_gb": 141,
        "hourly_cost_usd": 30.00,
        "memory_bandwidth_gb_s": 2952,
        "description": "Ultra-performance 8x H100 cluster for massive concurrency",
    },
    # Google TPUs
    "ct5lp-hightpu-1t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google TPU v5e (1-chip)",
        "accelerator_type": "TPU_V5_LITEPOD",
        "accelerator_count": 1,
        "vram_gb": 16,
        "hourly_cost_usd": 1.20,
        "memory_bandwidth_gb_s": 819,
        "description": "Cost-efficient TPU v5e single core",
    },
    "ct5lp-hightpu-4t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google TPU v5e (4-chip)",
        "accelerator_type": "TPU_V5_LITEPOD",
        "accelerator_count": 4,
        "vram_gb": 64,
        "hourly_cost_usd": 4.80,
        "memory_bandwidth_gb_s": 3276,
        "description": "TPU v5e 4-chip pod for medium open models",
    },
    "ct6e-standard-1t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (1-chip)",
        "accelerator_type": "TPU_V6E",
        "accelerator_count": 1,
        "vram_gb": 32,
        "hourly_cost_usd": 1.80,
        "memory_bandwidth_gb_s": 1640,
        "description": "Next-gen Trillium TPU optimized for low TPOT and high token throughput",
    },
    "ct6e-standard-4t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (4-chip)",
        "accelerator_type": "TPU_V6E",
        "accelerator_count": 4,
        "vram_gb": 128,
        "hourly_cost_usd": 7.20,
        "memory_bandwidth_gb_s": 6560,
        "description": "Trillium TPU 4-chip configuration for high-concurrency production serving",
    },
    "ct6e-standard-8t": {
        "family": AcceleratorFamily.TPU,
        "chip_name": "Google Trillium TPU v6e (8-chip)",
        "accelerator_type": "TPU_V6E",
        "accelerator_count": 8,
        "vram_gb": 256,
        "hourly_cost_usd": 14.40,
        "memory_bandwidth_gb_s": 13120,
        "description": "Trillium TPU 8-chip configuration for ultra-scale LLM inference",
    },
}

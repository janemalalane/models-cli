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
from typing import Any

# Google Pre-built Serving Container Images for Gemini Enterprise Online Prediction
DEFAULT_VLLM_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/agent-platform-mg-public/containers/pytorch-vllm-serve"
)
DEFAULT_SGLANG_CONTAINER_IMAGE = (
    "us-docker.pkg.dev/agent-platform-mg-public/containers/pytorch-sglang-serve"
)

DEFAULT_LLAMA_CONTAINER_IMAGE = ()


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


class OptimizationMetric(str, Enum):
    TTFT = "ttft"  # Time to First Token
    TPOT = "tpot"  # Time per Output Token
    THROUGHPUT = "throughput"  # Requests or Tokens per second
    COST = "cost"  # Price per hour / Price per token


class ModelOrigin(str, Enum):
    GOOGLE = "google"
    QWEN = "qwen"
    META = "meta"
    MISTRAL = "mistral"
    DEEPSEEK_AI = "deepseek-ai"
    MOONSHOTAI = "moonshotai"
    OPENAI = "openai"


class ModelArchitecture(str, Enum):
    DENSE = "dense"
    MOE = "moe"


class ModelQuantization(str, Enum):
    NONE = "none"
    FP8 = "fp8"
    GPTQ_4BIT = "gptq-4bit"
    MXFP4 = "mxfp4"


class ReasoningParser(str, Enum):
    GEMMA4 = "gemma4"
    QWEN3 = "qwen3"
    DEEPSEEK_R1 = "deepseek_r1"
    KIMI_K2 = "kimi_k2"
    OPENAI_GPTOSS = "openai_gptoss"


class ToolCallParser(str, Enum):
    GEMMA4 = "gemma4"
    HERMES = "hermes"
    QWEN3_CODER = "qwen3_coder"
    LLAMA3_JSON = "llama3_json"
    LLAMA4_PYTHONIC = "llama4_pythonic"
    MISTRAL = "mistral"
    DEEPSEEK_V3 = "deepseek_v3"
    KIMI_K2 = "kimi_k2"
    OPENAI = "openai"
    PYTHONIC = "pythonic"


DEFAULT_MODEL_REPO = "google/gemma-4-31B-it"
DEFAULT_REGION = "us-central1"
DEFAULT_ENGINE = "vllm"
DEFAULT_PRICING_MODEL = "on-demand"
GKE_RECOMMENDER_SERVICE = "gkerecommender.googleapis.com"


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
}

# Standard Open-Weights Model Catalog & Architecture Specifications
MODELS_SPECS: dict[str, dict[str, Any]] = {
    # Google (Gemma Series)
    "google/gemma-2-2b-it": {
        "model_name": "gemma-2-2b-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 2.0,
        "active_parameters_b": 2.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/google/gemma-2-2b-it",
    },
    "google/gemma-2-27b-it": {
        "model_name": "gemma-2-27b-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 27.0,
        "active_parameters_b": 27.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/google/gemma-2-27b-it",
    },
    "google/gemma-3-4b-it": {
        "model_name": "gemma-3-4b-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 4.0,
        "active_parameters_b": 4.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/google/gemma-3-4b-it",
    },
    "google/gemma-3-27b-it": {
        "model_name": "gemma-3-27b-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 27.0,
        "active_parameters_b": 27.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/google/gemma-3-27b-it",
    },
    "google/gemma-4-26B-A4B-it": {
        "model_name": "gemma-4-26B-A4B-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 25.2,
        "active_parameters_b": 3.8,
        "total_experts": 128,
        "activated_experts": 8,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.GEMMA4,
        "tool_call_parser": ToolCallParser.GEMMA4,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/google/gemma-4-26B-A4B-it",
    },
    "google/gemma-4-31B-it": {
        "model_name": "gemma-4-31B-it",
        "origin": ModelOrigin.GOOGLE,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 31.0,
        "active_parameters_b": 31.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.GEMMA4,
        "tool_call_parser": ToolCallParser.GEMMA4,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/google/gemma-4-31B-it",
    },
    # Qwen (Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 Series)
    "Qwen/Qwen3-8B-FP8": {
        "model_name": "Qwen3-8B-FP8",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 6.95,
        "active_parameters_b": 6.95,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.FP8,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.HERMES,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3-8B-FP8",
    },
    "Qwen/Qwen3-32B": {
        "model_name": "Qwen3-32B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 31.2,
        "active_parameters_b": 31.2,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.HERMES,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3-32B",
    },
    "Qwen/Qwen3-32B-FP8": {
        "model_name": "Qwen3-32B-FP8",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 31.2,
        "active_parameters_b": 31.2,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.FP8,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.HERMES,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3-32B-FP8",
    },
    "Qwen/Qwen3-235B-A22B": {
        "model_name": "Qwen3-235B-A22B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 234.0,
        "active_parameters_b": 22.0,
        "total_experts": 128,
        "activated_experts": 8,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.HERMES,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3-235B-A22B",
    },
    "Qwen/Qwen3-Coder-Next": {
        "model_name": "Qwen3-Coder-Next",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 79.0,
        "active_parameters_b": 3.0,
        "total_experts": 512,
        "activated_experts": 10,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3-Coder-Next",
    },
    "Qwen/Qwen3.5-9B": {
        "model_name": "Qwen3.5-9B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 9.0,
        "active_parameters_b": 9.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.5-9B",
    },
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4": {
        "model_name": "Qwen3.5-122B-A10B-GPTQ-Int4",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 122.0,
        "active_parameters_b": 10.0,
        "total_experts": 256,
        "activated_experts": 8,
        "shared_experts": 1,
        "quantization": ModelQuantization.GPTQ_4BIT,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    },
    "Qwen/Qwen3.5-397B-A17B": {
        "model_name": "Qwen3.5-397B-A17B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 397.0,
        "active_parameters_b": 17.0,
        "total_experts": 512,
        "activated_experts": 10,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.5-397B-A17B",
    },
    "Qwen/Qwen3.6-27B": {
        "model_name": "Qwen3.6-27B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 27.0,
        "active_parameters_b": 27.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.6-27B",
    },
    "Qwen/Qwen3.6-35B-A3B": {
        "model_name": "Qwen3.6-35B-A3B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 35.0,
        "active_parameters_b": 3.0,
        "total_experts": 256,
        "activated_experts": 8,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    },
    "Qwen/Qwen3.6-35B-A3B-FP8": {
        "model_name": "Qwen3.6-35B-A3B-FP8",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 35.0,
        "active_parameters_b": 3.0,
        "total_experts": 256,
        "activated_experts": 8,
        "shared_experts": 1,
        "quantization": ModelQuantization.FP8,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8",
    },
    "Qwen/Qwen3.8-27B": {
        "model_name": "Qwen3.8-27B",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 27.0,
        "active_parameters_b": 27.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.8-27B",
    },
    "Qwen/Qwen3.8-27B-FP8": {
        "model_name": "Qwen3.8-27B-FP8",
        "origin": ModelOrigin.QWEN,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 27.0,
        "active_parameters_b": 27.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.FP8,
        "reasoning_parser": ReasoningParser.QWEN3,
        "tool_call_parser": ToolCallParser.QWEN3_CODER,
        "gke_recommender_supported": False,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/Qwen/Qwen3.8-27B-FP8",
    },
    # Meta (Llama Series)
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "model_name": "Llama-3-8B",
        "origin": ModelOrigin.META,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 8.0,
        "active_parameters_b": 8.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_name": "Llama-3.2-1B-Instruct",
        "origin": ModelOrigin.META,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 1.0,
        "active_parameters_b": 1.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.LLAMA3_JSON,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "model_name": "Llama-3.3-70B-Instruct",
        "origin": ModelOrigin.META,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 70.0,
        "active_parameters_b": 70.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.LLAMA3_JSON,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "model_name": "Llama-4-Scout-17B-16E-Instruct",
        "origin": ModelOrigin.META,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 109.0,
        "active_parameters_b": 17.0,
        "total_experts": 16,
        "activated_experts": 2,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.LLAMA4_PYTHONIC,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "origin": ModelOrigin.META,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 400.0,
        "active_parameters_b": 17.0,
        "total_experts": 128,
        "activated_experts": 16,
        "shared_experts": 1,
        "quantization": ModelQuantization.FP8,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.LLAMA4_PYTHONIC,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
    # Mistral (Mistral & Mixtral Series)
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "model_name": "Mistral-Small-24B-Instruct-2501",
        "origin": ModelOrigin.MISTRAL,
        "architecture": ModelArchitecture.DENSE,
        "total_parameters_b": 24.0,
        "active_parameters_b": 24.0,
        "total_experts": None,
        "activated_experts": None,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.MISTRAL,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501",
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "model_name": "Mixtral-8x7B-Instruct-v0.1",
        "origin": ModelOrigin.MISTRAL,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 46.7,
        "active_parameters_b": 13.0,
        "total_experts": 8,
        "activated_experts": 2,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.MISTRAL,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "model_name": "Mixtral-8x22B-Instruct-v0.1",
        "origin": ModelOrigin.MISTRAL,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 141.0,
        "active_parameters_b": 39.0,
        "total_experts": 8,
        "activated_experts": 2,
        "shared_experts": None,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": None,
        "tool_call_parser": ToolCallParser.MISTRAL,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1",
    },
    # DeepSeek AI
    "deepseek-ai/DeepSeek-R1": {
        "model_name": "DeepSeek-R1",
        "origin": ModelOrigin.DEEPSEEK_AI,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 671.0,
        "active_parameters_b": 37.0,
        "total_experts": 257,
        "activated_experts": 256,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.DEEPSEEK_R1,
        "tool_call_parser": ToolCallParser.DEEPSEEK_V3,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1",
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model_name": "DeepSeek-R1-Distill-Qwen-7B",
        "origin": ModelOrigin.DEEPSEEK_AI,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 671.0,
        "active_parameters_b": 37.0,
        "total_experts": 257,
        "activated_experts": 256,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.DEEPSEEK_R1,
        "tool_call_parser": None,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    },
    # Moonshot AI
    "moonshotai/Kimi-K2.5": {
        "model_name": "Kimi-K2.5",
        "origin": ModelOrigin.MOONSHOTAI,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 1000.0,
        "active_parameters_b": 32.0,
        "total_experts": 384,
        "activated_experts": 8,
        "shared_experts": 1,
        "quantization": ModelQuantization.NONE,
        "reasoning_parser": ReasoningParser.KIMI_K2,
        "tool_call_parser": ToolCallParser.KIMI_K2,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/moonshotai/Kimi-K2.5",
    },
    # OpenAI (GPT-OSS Series)
    "openai/gpt-oss-20b": {
        "model_name": "gpt-oss-20b",
        "origin": ModelOrigin.OPENAI,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 21.0,
        "active_parameters_b": 3.6,
        "total_experts": 32,
        "activated_experts": 4,
        "shared_experts": None,
        "quantization": ModelQuantization.MXFP4,
        "reasoning_parser": ReasoningParser.OPENAI_GPTOSS,
        "tool_call_parser": ToolCallParser.OPENAI,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-10",
        "model_card_url": "https://huggingface.co/openai/gpt-oss-20b",
    },
    "openai/gpt-oss-120b": {
        "model_name": "gpt-oss-120b",
        "origin": ModelOrigin.OPENAI,
        "architecture": ModelArchitecture.MOE,
        "total_parameters_b": 117.0,
        "active_parameters_b": 5.1,
        "total_experts": 128,
        "activated_experts": 4,
        "shared_experts": None,
        "quantization": ModelQuantization.MXFP4,
        "reasoning_parser": ReasoningParser.OPENAI_GPTOSS,
        "tool_call_parser": ToolCallParser.OPENAI,
        "gke_recommender_supported": True,
        "rank": None,
        "timestamp": "2026-11-04",
        "model_card_url": "https://huggingface.co/openai/gpt-oss-120b",
    },
}


def lookup_model_spec(
    model_id: str,
    models_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Looks up a model in MODELS_SPECS by full repo ID or short model_name."""
    catalog = models_specs if models_specs is not None else MODELS_SPECS
    cleaned = model_id.strip()
    if not cleaned:
        return None

    if cleaned in catalog:
        return cleaned, catalog[cleaned]

    lowered = cleaned.lower()
    for repo_id, spec in catalog.items():
        if repo_id.lower() == lowered:
            return repo_id, spec

    for repo_id, spec in catalog.items():
        if str(spec.get("model_name", "")).lower() == lowered:
            return repo_id, spec

    return None


def get_vllm_spec_params(
    model_id: str | None,
    models_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return model-specific vLLM serving parameters derived from MODELS_SPECS.

    Includes ``served_model_name`` and, when defined on the catalog entry,
    ``reasoning_parser``, ``tool_call_parser``, and ``enable_auto_tool_choice``.
    """
    if not model_id or not model_id.strip():
        return {}

    cleaned = model_id.strip()
    spec_entry = lookup_model_spec(cleaned, models_specs=models_specs)
    served_name = spec_entry[0] if spec_entry is not None else cleaned
    model_spec = spec_entry[1] if spec_entry is not None else {}

    params: dict[str, Any] = {"served_model_name": served_name}

    rp = model_spec.get("reasoning_parser")
    if rp is not None:
        params["reasoning_parser"] = rp.value if hasattr(rp, "value") else str(rp)

    tcp = model_spec.get("tool_call_parser")
    if tcp is not None:
        params["tool_call_parser"] = tcp.value if hasattr(tcp, "value") else str(tcp)
        params["enable_auto_tool_choice"] = True

    return params

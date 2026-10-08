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

"""Compatibility hooks for inference-perf execution."""

# Fix upstream transformers bug where extra_special_tokens in tokenizer_config.json
# is a list rather than a dict (e.g. in google/diffusiongemma-26B-A4B-it).
try:
    import transformers.tokenization_utils_base as tub

    _orig = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def _safe_set(self, special_tokens=None, *args, **kwargs):
        if isinstance(special_tokens, (list, tuple, set)):
            special_tokens = {
                f"extra_token_{i}": t for i, t in enumerate(special_tokens)
            }
        return _orig(self, special_tokens, *args, **kwargs)

    tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _safe_set
except Exception:
    pass

#!/bin/sh
uv run main.py \
      --model_name Qwen/Qwen3.5-2B \
      --quantization ggml_q4km \
      --use_self_generated_data

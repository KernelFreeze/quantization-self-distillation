# Quantization Self-Distillation for LLMs

An experiment that finetunes a quantized (FP8/INT8/INT4 and GGML quants like q8_0 and q4_km) model to recover accuracy by distilling from the original FP16/BF16 model's outputs.

## Main hypothesis

1. Generate soft targets (logits) from the full-precision "teacher"
2. Train the quantized "student" to match those logits via KL divergence
3. The student learns to compensate for quantization rounding errors

## Usage

```
uv run main.py \
      --model_name Qwen/Qwen3.5-2B \
      --quantization ggml_q4km \
      --use_self_generated_data \
      --output_dir ./distilled_model
```

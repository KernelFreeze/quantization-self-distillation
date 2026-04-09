import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GGML Quantization Simulation (Q8_0, Q4_K_M) with STE
class STERound(torch.autograd.Function):
    """Straight-Through Estimator for rounding: forward rounds, backward passes gradient through."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def simulate_q8_0(weight: torch.Tensor) -> torch.Tensor:
    """
    Simulate GGML Q8_0 quantization (symmetric 8-bit, block size 32).

    Matches ggml-quants.c quantize_row_q8_0_ref / dequantize_row_q8_0:
      d = max(|x|) / 127  per block of 32
      qs = round(x / d)   clamped to [-128, 127]
      x_deq = d * qs
    """
    orig_shape = weight.shape
    flat = weight.reshape(orig_shape[0], -1)
    in_f = flat.shape[1]

    # Pad to multiple of 32
    pad = (32 - in_f % 32) % 32
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    # Reshape into blocks of 32
    blocked = flat.reshape(-1, 32)  # (num_blocks, 32)

    # Per-block scale: d = amax / 127
    amax = blocked.abs().amax(dim=-1, keepdim=True)  # (num_blocks, 1)
    d = amax / 127.0
    # Avoid division by zero for all-zero blocks
    d_safe = torch.where(d == 0, torch.ones_like(d), d)

    # Quantize with STE, clamp to int8 range
    qs = torch.clamp(STERound.apply(blocked / d_safe), -128, 127)

    # Dequantize
    w_deq = d * qs

    # Reshape back and remove padding
    w_deq = w_deq.reshape(orig_shape[0], -1)
    if pad > 0:
        w_deq = w_deq[:, :in_f]
    return w_deq.reshape(orig_shape)


def simulate_q4_k_m(weight: torch.Tensor) -> torch.Tensor:
    """
    Simulate GGML Q4_K_M quantization (asymmetric 4-bit, super-block 256 = 8 sub-blocks of 32).

    Matches ggml-quants.c quantize_row_q4_K_ref / dequantize_row_q4_K:
      Per sub-block (32 elems): find min, range -> sc = range/15, m = -min
      Per super-block (8 sub-blocks): d = max(sc)/63, dmin = max(m)/63
      Quantize scales/mins to 6-bit, values to 4-bit unsigned [0,15]
      Dequant: y = d*sc_q*val_q - dmin*m_q
    """
    orig_shape = weight.shape
    flat = weight.reshape(orig_shape[0], -1)
    in_f = flat.shape[1]

    # Pad to multiple of 256
    pad = (256 - in_f % 256) % 256
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    # Reshape: (out_f, in_f_padded) -> (N, 8, 32) where N = out_f * num_superblocks
    w = flat.reshape(-1, 8, 32)

    # Per sub-block min and range
    sub_min = w.min(dim=-1, keepdim=True).values  # (N, 8, 1)
    sub_max = w.max(dim=-1, keepdim=True).values  # (N, 8, 1)
    sub_range = sub_max - sub_min

    sc = sub_range / 15.0  # sub-block scale
    m = -sub_min  # sub-block minimum offset (positive)

    # Super-block: quantize scales and mins to 6-bit
    max_sc = sc.amax(dim=1, keepdim=True)  # (N, 1, 1)
    max_m = m.amax(dim=1, keepdim=True)  # (N, 1, 1)

    d = max_sc / 63.0  # super-block scale-of-scales
    dmin = max_m / 63.0  # super-block scale-of-mins

    d_safe = torch.where(d == 0, torch.ones_like(d), d)
    dmin_safe = torch.where(dmin == 0, torch.ones_like(dmin), dmin)

    # STE round sub-block scales/mins to 6-bit integers [0, 63]
    sc_q = torch.clamp(STERound.apply(sc / d_safe), 0, 63)
    m_q = torch.clamp(STERound.apply(m / dmin_safe), 0, 63)

    # Effective (dequantized) sub-block scale and min
    sc_eff = d * sc_q  # (N, 8, 1)
    m_eff = dmin * m_q  # (N, 8, 1)

    # Quantize values to 4-bit unsigned [0, 15]
    sc_eff_safe = torch.where(sc_eff == 0, torch.ones_like(sc_eff), sc_eff)
    val_q = torch.clamp(STERound.apply((w + m_eff) / sc_eff_safe), 0, 15)

    # Dequantize
    w_deq = sc_eff * val_q - m_eff  # (N, 8, 32)

    # Reshape back and remove padding
    w_deq = w_deq.reshape(orig_shape[0], -1)
    if pad > 0:
        w_deq = w_deq[:, :in_f]
    return w_deq.reshape(orig_shape)


# GGML Quantized Linear Layers
class GGMLQuantizedLinear(torch.nn.Linear):
    """
    Linear layer with GGML quantization simulation on forward pass.
    Subclasses nn.Linear so PEFT dispatch_default recognizes it for LoRA wrapping.
    Weights stored in bf16; quantization noise is simulated via STE.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_type: str = "q8_0",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.quant_type = quant_type
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_type == "q8_0":
            w_deq = simulate_q8_0(self.weight)
        elif self.quant_type == "q4_k_m":
            w_deq = simulate_q4_k_m(self.weight)
        else:
            w_deq = self.weight
        return F.linear(x, w_deq, self.bias)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, quant_type: str = "q8_0"):
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quant_type=quant_type,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        layer.weight = linear.weight
        layer.weight.requires_grad_(False)
        if linear.bias is not None:
            layer.bias = linear.bias
        return layer


class AdaptiveGGMLLinear(torch.nn.Linear):
    """
    Linear layer that learns per-layer Q4_K_M vs Q8_0 preference.
    Uses sigmoid interpolation during training; hard assignment at inference.
    """

    Q8_BPW = 8.5  # bits per weight for Q8_0
    Q4_BPW = 4.5  # bits per weight for Q4_K_M

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.weight.requires_grad_(False)
        # Learnable: positive -> prefer Q8_0, negative -> prefer Q4_K_M
        self.bit_preference = torch.nn.Parameter(
            torch.tensor(0.0, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q8 = simulate_q8_0(self.weight)
        w_q4 = simulate_q4_k_m(self.weight)

        if self.training:
            prob_q8 = torch.sigmoid(self.bit_preference)
            w_deq = prob_q8 * w_q8 + (1.0 - prob_q8) * w_q4
        else:
            w_deq = w_q8 if self.bit_preference.item() > 0 else w_q4

        return F.linear(x, w_deq, self.bias)

    def effective_bits_per_weight(self) -> torch.Tensor:
        """Differentiable expected bits/weight for regularization."""
        prob_q8 = torch.sigmoid(self.bit_preference)
        return prob_q8 * self.Q8_BPW + (1.0 - prob_q8) * self.Q4_BPW

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear):
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        layer.weight = linear.weight
        layer.weight.requires_grad_(False)
        if linear.bias is not None:
            layer.bias = linear.bias
        return layer


def replace_linear_with_ggml(
    model: torch.nn.Module,
    quant_type: str,
    target_modules: Optional[list[str]] = None,
) -> torch.nn.Module:
    """Replace nn.Linear layers with GGML quantized variants."""
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Build a mapping of parent_fqn -> {child_name: new_module}
    replacements: dict[str, dict[str, torch.nn.Module]] = {}
    for fqn, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(t in fqn for t in target_modules):
            continue
        if "lm_head" in fqn or "embed" in fqn:
            continue

        parent_fqn, child_name = fqn.rsplit(".", 1)
        if quant_type == "ggml_adaptive":
            new_layer = AdaptiveGGMLLinear.from_linear(module)
        elif quant_type == "ggml_q8":
            new_layer = GGMLQuantizedLinear.from_linear(module, "q8_0")
        elif quant_type == "ggml_q4km":
            new_layer = GGMLQuantizedLinear.from_linear(module, "q4_k_m")
        else:
            continue

        replacements.setdefault(parent_fqn, {})[child_name] = new_layer

    # Apply replacements
    module_dict = dict(model.named_modules())
    for parent_fqn, children in replacements.items():
        parent = module_dict[parent_fqn]
        for child_name, new_layer in children.items():
            setattr(parent, child_name, new_layer)

    return model


def compute_bit_budget_loss(model: torch.nn.Module) -> torch.Tensor:
    """Average effective bits/weight across all AdaptiveGGMLLinear layers (differentiable)."""
    total_bits = torch.tensor(0.0, device=next(model.parameters()).device)
    total_params = 0

    for module in model.modules():
        if isinstance(module, AdaptiveGGMLLinear):
            n = module.weight.numel()
            total_bits = total_bits + module.effective_bits_per_weight() * n
            total_params += n

    if total_params == 0:
        return total_bits
    return total_bits / total_params


def print_bit_allocation(model: torch.nn.Module):
    """Log per-layer bit allocation decisions after training."""
    logger.info("=" * 60)
    logger.info("BIT ALLOCATION (per layer)")
    logger.info("=" * 60)

    total_bits = 0.0
    total_params = 0

    for fqn, module in model.named_modules():
        if isinstance(module, AdaptiveGGMLLinear):
            prob_q8 = torch.sigmoid(module.bit_preference).item()
            chosen = "Q8_0" if module.bit_preference.item() > 0 else "Q4_K_M"
            bpw = module.effective_bits_per_weight().item()
            n = module.weight.numel()
            total_bits += bpw * n
            total_params += n
            logger.info(
                f"  {fqn:50s} | {chosen:6s} | prob_q8={prob_q8:.3f} | bpw={bpw:.2f}"
            )

    if total_params > 0:
        avg_bpw = total_bits / total_params
        logger.info(f"  {'AVERAGE':50s} | {'':6s} | {'':14s} | bpw={avg_bpw:.2f}")
    logger.info("=" * 60)


# STEP 1: Quantization Configurations
def get_quantization_config(quant_type: str) -> Optional[BitsAndBytesConfig]:
    """
    Return a BitsAndBytesConfig for the desired quantization level.

    FP8 is handled separately via torchao.float8 (not bitsandbytes),
    so this returns None for fp8 — see load_student_model().
    """
    configs = {
        "int4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 works best
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # nested quantization saves memory
        ),
        "int8": BitsAndBytesConfig(
            load_in_8bit=True,
        ),
        "fp8": None,  # Handled via torchao.float8 in load_student_model()
        "ggml_q8": None,  # Handled via GGML simulation in load_student_model()
        "ggml_q4km": None,
        "ggml_adaptive": None,
        "none": None,
    }
    if quant_type not in configs:
        raise ValueError(
            f"Unknown quantization type: {quant_type}. Choose from {list(configs.keys())}"
        )
    return configs[quant_type]


def load_teacher_model(model_name: str, device: str = "auto"):
    """Load the full-precision teacher model (FP16/BF16)."""
    logger.info(f"Loading teacher model: {model_name} in full precision")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()  # Teacher is frozen — never trains
    return model


def _fp8_module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
    """Only convert nn.Linear layers whose dimensions are divisible by 16."""
    if isinstance(mod, torch.nn.Linear):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    return True


def load_student_model(model_name: str, quant_type: str, device: str = "auto"):
    """Load the quantized student model with LoRA adapters for training."""
    logger.info(f"Loading student model: {model_name} with {quant_type} quantization")

    quant_config = get_quantization_config(quant_type)

    if quant_type == "fp8":
        # True FP8 via torchao: load in bfloat16, then convert linear layers to FP8
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        fp8_config = Float8LinearConfig.from_recipe_name("rowwise")
        convert_to_float8_training(
            model,
            config=fp8_config,
            module_filter_fn=_fp8_module_filter_fn,
        )
        logger.info("Converted model to FP8 training via torchao.float8")
    elif quant_type.startswith("ggml"):
        # GGML quantization simulation: load in bf16, replace linears with GGML layers
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        model = replace_linear_with_ggml(model, quant_type, target_modules)
        logger.info(f"Converted model to {quant_type} quantization simulation")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device,
            trust_remote_code=True,
        )
        # Prepare quantized model for training (freezes base, enables grad on adapters)
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters — these are the trainable parameters
    # The base quantized weights stay frozen; LoRA learns to compensate
    lora_config = LoraConfig(
        r=64,  # Rank — higher = more capacity to recover
        lora_alpha=128,  # Scaling factor (alpha/r = effective LR multiplier)
        target_modules=[  # Which layers get adapters
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # For adaptive GGML: unfreeze bit_preference params (PEFT freezes all base params)
    if quant_type == "ggml_adaptive":
        for name, param in model.named_parameters():
            if "bit_preference" in name:
                param.requires_grad_(True)

    model.print_trainable_parameters()  # Should be ~1-2% of total
    return model


# STEP 2: Data Preparation
def prepare_dataset(tokenizer, max_length: int = 1024, num_samples: int = 10000):
    """
    Prepare training data. You have two options:

    Option A (shown here): Use a public text dataset.
        Pros: Simple, diverse text
        Cons: Distribution may not perfectly match model's pretraining data

    Option B (recommended for best results): Generate data FROM the teacher.
        See generate_self_data() below.
    """
    logger.info("Loading dataset...")

    # Using a diverse text dataset as training inputs
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Filter out empty strings and very short texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 100)

    # Take a subset — you don't need much data for distillation
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


def generate_self_data(
    teacher_model,
    tokenizer,
    num_samples: int = 5000,
    max_length: int = 512,
    batch_size: int = 4,
):
    """
    Option B: Generate training data from the teacher model itself.
    This is the approach used by LLM-QAT and tends to work better because
    the data distribution matches what the model "expects" to see.
    """
    logger.info(f"Generating {num_samples} samples from teacher model...")
    teacher_model.eval()

    all_input_ids = []

    # Build a minimal BOS prompt as input_ids directly to avoid empty tokenization
    bos_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    bos_ids = torch.tensor([[bos_id]] * batch_size, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            inputs = {
                "input_ids": bos_ids.to(teacher_model.device),
                "attention_mask": torch.ones_like(bos_ids).to(teacher_model.device),
            }

            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )
            all_input_ids.append(outputs.cpu())

            if (i // batch_size) % 50 == 0:
                logger.info(
                    f"  Generated {min(i + batch_size, num_samples)}/{num_samples}"
                )

    return torch.cat(all_input_ids, dim=0)[:num_samples]


class CachedLogitsDataset:
    """Wraps a dataset with pre-computed top-k teacher logits."""

    def __init__(self, dataset, top_k_values, top_k_indices):
        self.dataset = dataset
        self.top_k_values = top_k_values
        self.top_k_indices = top_k_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "top_k_values": self.top_k_values[idx],
            "top_k_indices": self.top_k_indices[idx],
        }


def precompute_teacher_logits(
    teacher_model,
    dataset,
    batch_size: int = 4,
    top_k: int = 128,
):
    """
    Pre-compute and cache top-k teacher logits for all samples.

    Stores only the top-k logit values and their indices per position,
    reducing memory from O(vocab_size) to O(k) per token. The resulting
    tensors are kept on CPU so VRAM can be freed after this completes.
    """
    logger.info(f"Pre-computing teacher logits (top-k={top_k})...")
    teacher_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_top_k_values = []
    all_top_k_indices = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(teacher_model.device)
            attention_mask = batch["attention_mask"].to(teacher_model.device)

            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, vocab)

            values, indices = logits.topk(top_k, dim=-1)  # (batch, seq_len, k)

            all_top_k_values.append(values.cpu().to(torch.bfloat16))
            all_top_k_indices.append(indices.cpu().to(torch.int32))

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Processed {(i + 1) * batch_size}/{len(dataset)} samples"
                )

    top_k_values = torch.cat(all_top_k_values, dim=0)
    top_k_indices = torch.cat(all_top_k_indices, dim=0)

    size_gb = (top_k_values.nbytes + top_k_indices.nbytes) / (1024**3)
    logger.info(f"Cached teacher logits: {top_k_values.shape}, size: {size_gb:.2f} GB")

    return top_k_values, top_k_indices


# STEP 3: Distillation Loss
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7,
):
    """
    Combined distillation loss:
      L = alpha * KL_div(student, teacher)  +  (1 - alpha) * CE(student, labels)

    Args:
        student_logits: [batch, seq_len, vocab_size] from quantized model
        teacher_logits: [batch, seq_len, vocab_size] from full-precision model
        labels:         [batch, seq_len] ground truth token IDs
        temperature:    Softens probability distributions. Higher = softer.
                        T=2-4 is typical. Higher preserves more info from the tail.
        alpha:          Balance between KL loss and hard-label CE loss.
                        0.7 = mostly distillation, 0.3 = hard labels as regularizer.
    """
    vocab_size = student_logits.size(-1)

    #  KL Divergence Loss (soft targets) ---
    # Scale logits by temperature before softmax
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student) — note: input is log-probs, target is probs
    kl_loss = F.kl_div(
        student_soft.view(-1, vocab_size),
        teacher_soft.view(-1, vocab_size),
        reduction="batchmean",
    )
    # Scale by T^2 to make gradients comparable across temperatures
    kl_loss = kl_loss * (temperature**2)

    # CrossEntropy Loss (hard targets) ---
    # This acts as a regularizer to keep the model grounded
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # Combined
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return total_loss, kl_loss.item(), ce_loss.item()


def distillation_loss_from_topk(
    student_logits: torch.Tensor,
    top_k_values: torch.Tensor,
    top_k_indices: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7,
):
    """
    Distillation loss using cached top-k teacher logits.

    Reconstructs an approximate teacher distribution by placing the cached
    top-k logit values at their original positions and filling the rest
    with -1e9 (effectively zero probability after softmax). This closely
    approximates the full KL divergence since the tail contributes
    negligible probability mass.
    """
    vocab_size = student_logits.size(-1)

    # Reconstruct approximate teacher logits from top-k
    teacher_logits = torch.full_like(student_logits, -1e9)
    tk_values = top_k_values.to(
        dtype=student_logits.dtype, device=student_logits.device
    )
    tk_indices = top_k_indices.to(dtype=torch.long, device=student_logits.device)
    teacher_logits.scatter_(-1, tk_indices, tk_values)

    #  KL Divergence Loss (soft targets) ---
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_soft.view(-1, vocab_size),
        teacher_soft.view(-1, vocab_size),
        reduction="batchmean",
    )
    kl_loss = kl_loss * (temperature**2)

    # CrossEntropy Loss (hard targets) ---
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return total_loss, kl_loss.item(), ce_loss.item()


# STEP 4: Training Loop
def train(
    teacher_model,
    student_model,
    tokenizer,
    dataset,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    temperature: float = 2.0,
    alpha: float = 0.7,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    output_dir: str = "./distilled_model",
    bit_budget_lambda: float = 0.0,
    quant_type: str = "",
):
    """Main distillation training loop."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Separate param groups: LoRA params + bit_preference params (higher LR)
    lora_params = [
        p
        for n, p in student_model.named_parameters()
        if p.requires_grad and "bit_preference" not in n
    ]
    bit_params = [
        p
        for n, p in student_model.named_parameters()
        if p.requires_grad and "bit_preference" in n
    ]
    param_groups = [{"params": lora_params, "lr": learning_rate}]
    if bit_params:
        param_groups.append({"params": bit_params, "lr": learning_rate * 10})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    logger.info("Starting distillation training:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(
        f"  Batch size: {batch_size} x {gradient_accumulation_steps} accumulation"
    )
    logger.info(f"  Total optimization steps: {total_steps}")
    logger.info(f"  Temperature: {temperature}, Alpha: {alpha}")

    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()

    global_step = 0
    for epoch in range(epochs):
        total_loss_accum = 0
        total_kl_accum = 0
        total_ce_accum = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(student_model.device)
            attention_mask = batch["attention_mask"].to(student_model.device)

            # Get student logits
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits

            # Compute distillation loss
            if "top_k_values" in batch:
                # Use pre-computed cached teacher logits
                loss, kl_val, ce_val = distillation_loss_from_topk(
                    student_logits=student_logits,
                    top_k_values=batch["top_k_values"],
                    top_k_indices=batch["top_k_indices"],
                    labels=input_ids,
                    temperature=temperature,
                    alpha=alpha,
                )
            else:
                # Online teacher forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_logits = teacher_outputs.logits.detach()

                loss, kl_val, ce_val = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=input_ids,
                    temperature=temperature,
                    alpha=alpha,
                )

            # Bit-budget regularization for adaptive GGML quantization
            if bit_budget_lambda > 0 and quant_type == "ggml_adaptive":
                bit_loss = compute_bit_budget_loss(student_model)
                loss = loss + bit_budget_lambda * bit_loss

            # Scale for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_loss_accum += loss.item()
            total_kl_accum += kl_val
            total_ce_accum += ce_val

            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = total_loss_accum / 50
                    avg_kl = total_kl_accum / (50 * gradient_accumulation_steps)
                    avg_ce = total_ce_accum / (50 * gradient_accumulation_steps)
                    msg = (
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | CE: {avg_ce:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    if quant_type == "ggml_adaptive":
                        avg_bpw = compute_bit_budget_loss(student_model).item()
                        msg += f" | bpw: {avg_bpw:.2f}"
                    logger.info(msg)
                    total_loss_accum = 0
                    total_kl_accum = 0
                    total_ce_accum = 0

        logger.info(f"Epoch {epoch + 1}/{epochs} complete")

    # Log bit allocation for adaptive mode
    if quant_type == "ggml_adaptive":
        print_bit_allocation(student_model)


# STEP 5: Save Student Model
def save_model(
    student_model,
    tokenizer,
    output_dir: str,
    save_adapters: bool = True,
    save_merged: bool = False,
):
    """
    Save the distilled student model.

    Args:
        student_model: The PEFT-wrapped student model after training.
        tokenizer: The tokenizer to save alongside the model.
        output_dir: Base output directory.
        save_adapters: If True, save LoRA adapters separately (small, can be
                       loaded on top of the base model later).
        save_merged: If True, merge LoRA weights into the base model and save
                     the full merged model. This is larger but standalone.
    """
    import os

    if save_adapters:
        adapter_dir = os.path.join(output_dir, "adapters")
        student_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info(f"LoRA adapters saved to {adapter_dir}")

    if save_merged:
        merged_dir = os.path.join(output_dir, "merged")
        logger.info("Merging LoRA weights into base model...")
        merged_model = student_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged model saved to {merged_dir}")

    if not save_adapters and not save_merged:
        logger.warning("No save option selected — model was not saved.")


# STEP 6: Evaluation — Compare Before/After
@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_samples=200,
):
    """Compute perplexity on a held-out set to measure recovery."""
    model.eval()
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 50)
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    total_loss = 0
    total_tokens = 0

    for example in dataset:
        inputs = tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        outputs = model(**inputs, labels=inputs["input_ids"])
        num_tokens = inputs["input_ids"].numel()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Quantization Self-Distillation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int4", "int8", "fp8", "ggml_q8", "ggml_q4km", "ggml_adaptive"],
    )
    parser.add_argument("--output_dir", type=str, default="./distilled_quantized_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=16,
        help="Batch size for self-generated data (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device map for model loading (default: cuda)",
    )
    parser.add_argument(
        "--use_self_generated_data",
        action="store_true",
        help="Generate training data from teacher model (recommended)",
    )
    parser.add_argument(
        "--bit_budget_lambda",
        type=float,
        default=0.01,
        help="Regularization weight for bit-budget loss (ggml_adaptive only)",
    )
    parser.add_argument(
        "--save_merged",
        action="store_true",
        help="Merge LoRA weights into the base model and save the full merged model",
    )
    parser.add_argument(
        "--no_save_adapters",
        action="store_true",
        help="Skip saving LoRA adapters separately (use with --save_merged)",
    )
    parser.add_argument(
        "--offload_teacher",
        action="store_true",
        help="Unload teacher after caching logits to reduce VRAM (only one model loaded at a time)",
    )
    parser.add_argument(
        "--top_k_logits",
        type=int,
        default=128,
        help="Number of top-k logits to cache per token when using --offload_teacher (default: 128)",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Helper to build a dataset from self-generated or public data
    class DictDataset:
        def __init__(self, ids, mask):
            self.ids = ids
            self.mask = mask

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return {"input_ids": self.ids[idx], "attention_mask": self.mask[idx]}

    def _prepare_data(teacher_model):
        if args.use_self_generated_data:
            generated_ids = generate_self_data(
                teacher_model,
                tokenizer,
                num_samples=args.num_samples,
                max_length=args.max_length,
                batch_size=args.gen_batch_size,
            )
            attention_mask = (generated_ids != tokenizer.pad_token_id).long()
            return DictDataset(generated_ids, attention_mask)
        else:
            return prepare_dataset(tokenizer, args.max_length, args.num_samples)

    # Two flows: offload_teacher (sequential) vs default (both in VRAM)
    if args.offload_teacher:
        # Sequential flow: only one model in VRAM at a time
        import gc

        # Phase 1: Teacher — evaluate, generate data, cache logits
        teacher = load_teacher_model(args.model_name, device=args.device)
        teacher = torch.compile(teacher)

        logger.info("Evaluating teacher (FP16) perplexity...")
        teacher_ppl = evaluate_perplexity(teacher, tokenizer)
        logger.info(f"  Teacher perplexity: {teacher_ppl:.2f}")

        dataset = _prepare_data(teacher)

        top_k_values, top_k_indices = precompute_teacher_logits(
            teacher,
            dataset,
            batch_size=args.batch_size,
            top_k=args.top_k_logits,
        )

        # Unload teacher to free VRAM
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Teacher model offloaded — VRAM freed for student")

        dataset = CachedLogitsDataset(dataset, top_k_values, top_k_indices)
        teacher = None  # train() won't use teacher forward passes

        # Phase 2: Student — evaluate, train, evaluate
        student = load_student_model(
            args.model_name, args.quantization, device=args.device
        )

        logger.info("Evaluating student (quantized, before distillation) perplexity...")
        student_ppl_before = evaluate_perplexity(student, tokenizer)
        logger.info(f"  Student perplexity (before): {student_ppl_before:.2f}")
    else:
        # Default flow: both models in VRAM
        teacher = load_teacher_model(args.model_name, device=args.device)
        teacher = torch.compile(teacher)
        student = load_student_model(
            args.model_name, args.quantization, device=args.device
        )

        logger.info("Evaluating teacher (FP16) perplexity...")
        teacher_ppl = evaluate_perplexity(teacher, tokenizer)
        logger.info(f"  Teacher perplexity: {teacher_ppl:.2f}")

        logger.info("Evaluating student (quantized, before distillation) perplexity...")
        student_ppl_before = evaluate_perplexity(student, tokenizer)
        logger.info(f"  Student perplexity (before): {student_ppl_before:.2f}")

        dataset = _prepare_data(teacher)

    # Distillation Training
    train(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        output_dir=args.output_dir,
        bit_budget_lambda=args.bit_budget_lambda,
        quant_type=args.quantization,
    )

    # Save the distilled model
    save_model(
        student_model=student,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        save_adapters=not args.no_save_adapters,
        save_merged=args.save_merged,
    )

    # Evaluate AFTER distillation
    logger.info("Evaluating student (after distillation) perplexity...")
    student_ppl_after = evaluate_perplexity(student, tokenizer)
    logger.info(f"  Student perplexity (after): {student_ppl_after:.2f}")

    # Summary
    gap_before = student_ppl_before - teacher_ppl
    gap_after = student_ppl_after - teacher_ppl
    recovery = (1 - gap_after / gap_before) * 100 if gap_before > 0 else 0

    logger.info("=" * 60)
    logger.info("DISTILLATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Teacher (FP16) perplexity:     {teacher_ppl:.2f}")
    logger.info(
        f"  Student BEFORE distillation:   {student_ppl_before:.2f}  (gap: +{gap_before:.2f})"
    )
    logger.info(
        f"  Student AFTER distillation:    {student_ppl_after:.2f}  (gap: +{gap_after:.2f})"
    )
    logger.info(f"  Accuracy recovered:            {recovery:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

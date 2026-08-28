#!/usr/bin/env python3
"""D5: Export MiniGPT Wq sub-tiles for int8_linear_layer RTL regression.

Loads blocks.0.attention.qkv_proj Q weights from a MiniGPT checkpoint, captures
norm1 activations, quantizes with int8_gemm_reference_v2 (shift_bits=24), and
returns D4-compatible case dicts (A / WEIGHT / MULT / EXPECT as INT8 lists).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SHIFT_BITS = 24
MULT_MAX = (1 << 18) - 1
MULT_MIN = 1

DEFAULT_LEARN_ROOT = Path(r"F:/Users/22563/Desktop/learn")
DEFAULT_CHECKPOINT = DEFAULT_LEARN_ROOT / "mini_gpt_ready" / "mini_gpt_checkpoint.pt"
DEFAULT_MINI_GPT_MODELS = DEFAULT_LEARN_ROOT / "mini_gpt_ready" / "models"
DEFAULT_REF_DIR = DEFAULT_LEARN_ROOT / "int8_gemm_reference"


@dataclass(frozen=True)
class SliceSpec:
    case_id: int
    m: int
    n: int
    k: int
    m_row_start: int
    n_out_start: int
    k_in_start: int
    a_gap: int = 0
    c_hold: int = 2


def _setup_import_paths(
    learn_root: Path,
    mini_gpt_models: Path,
    ref_dir: Path,
) -> None:
    for path in (mini_gpt_models, ref_dir):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_reference():
    from int8_gemm_reference import (  # noqa: WPS433
        int8_gemm_int32,
        quantize_per_tensor,
        quantize_weight_per_output_channel,
        symmetric_scale,
    )
    from int8_gemm_reference_v2 import requantize_accumulator_fixed  # noqa: WPS433

    return (
        quantize_per_tensor,
        quantize_weight_per_output_channel,
        int8_gemm_int32,
        requantize_accumulator_fixed,
        symmetric_scale,
    )


def weight_rtl_flat(weight_q: np.ndarray) -> list[int]:
    """Convert Wq[j,k] INT8 array [N,K] to RTL column-major K×N flat."""
    weight_q = np.asarray(weight_q, dtype=np.int8)
    n_out, k_in = weight_q.shape
    values: list[int] = []
    for kk in range(k_in):
        for j in range(n_out):
            values.append(int(weight_q[j, kk]))
    return values


def activation_rtl_flat(input_q: np.ndarray) -> list[int]:
    """Row-major M×K flat list."""
    return [int(v) for v in np.asarray(input_q, dtype=np.int8).reshape(-1)]


def expected_rtl_flat(output_q: np.ndarray) -> list[int]:
    """Row-major M×N flat list."""
    return [int(v) for v in np.asarray(output_q, dtype=np.int8).reshape(-1)]


def build_case_specs() -> list[SliceSpec]:
    specs: list[SliceSpec] = []

    for idx, n_out_start in enumerate((0, 4, 8, 12)):
        specs.append(
            SliceSpec(
                case_id=idx + 1,
                m=4,
                n=4,
                k=16,
                m_row_start=0,
                n_out_start=n_out_start,
                k_in_start=0,
            )
        )

    for idx, m_row_start in enumerate((0, 4, 8, 12)):
        specs.append(
            SliceSpec(
                case_id=idx + 5,
                m=4,
                n=4,
                k=16,
                m_row_start=m_row_start,
                n_out_start=0,
                k_in_start=0,
            )
        )

    specs.extend(
        [
            SliceSpec(9, 2, 2, 8, 0, 0, 0),
            SliceSpec(10, 3, 3, 12, 2, 4, 4),
            SliceSpec(11, 4, 2, 16, 0, 8, 0),
            SliceSpec(12, 2, 4, 16, 6, 0, 16, c_hold=3),
        ]
    )
    return specs


def capture_block0_norm1_activation(
    checkpoint_path: Path,
    mini_gpt_models: Path,
    seq_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from mini_gpt import MiniGPT  # noqa: WPS433

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ckpt["model_config"]
    stoi = ckpt["stoi"]

    model = MiniGPT(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["norm1_out"] = output.detach().cpu()

    handle = model.blocks[0].norm1.register_forward_hook(hook)

    rng = np.random.default_rng(seed)
    text_path = mini_gpt_models.parent / "data" / "input.txt"
    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
        start = int(rng.integers(0, max(1, len(text) - seq_len)))
        chunk = text[start : start + seq_len]
        token_ids = [stoi[c] for c in chunk if c in stoi]
        while len(token_ids) < seq_len:
            token_ids.append(next(iter(stoi.values())))
        token_ids = token_ids[:seq_len]
    else:
        token_ids = rng.integers(0, model_config["vocab_size"], size=seq_len).tolist()

    tokens = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        model(tokens)

    handle.remove()

    activation = captured["norm1_out"].numpy()[0].astype(np.float32)
    qkv_weight = ckpt["model_state_dict"]["blocks.0.attention.qkv_proj.weight"].numpy()
    wq = qkv_weight[0:64, :].astype(np.float32)
    return activation, wq


def export_case(
    spec: SliceSpec,
    activation_fp32: np.ndarray,
    wq_fp32: np.ndarray,
    ref_fns,
) -> dict[str, Any]:
    (
        quantize_per_tensor,
        quantize_weight_per_output_channel,
        int8_gemm_int32,
        requantize_accumulator_fixed,
        symmetric_scale,
    ) = ref_fns

    m_end = spec.m_row_start + spec.m
    n_end = spec.n_out_start + spec.n
    k_end = spec.k_in_start + spec.k

    if m_end > activation_fp32.shape[0]:
        raise ValueError(f"case {spec.case_id}: m slice exceeds activation rows")
    if n_end > wq_fp32.shape[0]:
        raise ValueError(f"case {spec.case_id}: n slice exceeds Wq rows")
    if k_end > wq_fp32.shape[1]:
        raise ValueError(f"case {spec.case_id}: k slice exceeds Wq cols")

    a_fp32 = activation_fp32[spec.m_row_start:m_end, spec.k_in_start:k_end]
    w_fp32 = wq_fp32[spec.n_out_start:n_end, spec.k_in_start:k_end]

    input_q, input_scale, _ = quantize_per_tensor(a_fp32)
    weight_q, weight_scales, _ = quantize_weight_per_output_channel(w_fp32)
    accumulator = int8_gemm_int32(input_q, weight_q)

    baseline_fp32 = a_fp32 @ w_fp32.T
    output_scale = symmetric_scale(float(np.max(np.abs(baseline_fp32))))

    (
        output_q,
        _sat_count,
        _real_mult,
        integer_multipliers,
        _pre_clip,
    ) = requantize_accumulator_fixed(
        accumulator,
        input_scale,
        weight_scales,
        output_scale,
        SHIFT_BITS,
    )

    mults = [int(v) for v in integer_multipliers.tolist()]
    for mult in mults:
        if mult < MULT_MIN or mult > MULT_MAX:
            raise ValueError(
                f"case {spec.case_id}: multiplier {mult} out of 18-bit range "
                f"[{MULT_MIN}, {MULT_MAX}]"
            )

    return {
        "case_id": spec.case_id,
        "m": spec.m,
        "n": spec.n,
        "k": spec.k,
        "a_gap": spec.a_gap,
        "c_hold": spec.c_hold,
        "a": activation_rtl_flat(input_q),
        "weights": weight_rtl_flat(weight_q),
        "mults": mults,
        "expected": expected_rtl_flat(output_q),
        "meta": {
            "m_row_start": spec.m_row_start,
            "n_out_start": spec.n_out_start,
            "k_in_start": spec.k_in_start,
            "input_scale": float(input_scale),
            "weight_scales": [float(v) for v in weight_scales.tolist()],
            "output_scale": float(output_scale),
        },
    }


def export_all_cases(
    checkpoint_path: Path,
    learn_root: Path,
    seq_len: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mini_gpt_models = learn_root / "mini_gpt_ready" / "models"
    ref_dir = learn_root / "int8_gemm_reference"
    _setup_import_paths(learn_root, mini_gpt_models, ref_dir)
    ref_fns = _load_reference()

    activation_fp32, wq_fp32 = capture_block0_norm1_activation(
        checkpoint_path, mini_gpt_models, seq_len, seed
    )

    cases: list[dict[str, Any]] = []
    for spec in build_case_specs():
        cases.append(export_case(spec, activation_fp32, wq_fp32, ref_fns))

    header = {
        "source": "mini_gpt blocks.0.attention.qkv_proj Q slice",
        "checkpoint": str(checkpoint_path.resolve()),
        "seq_len": seq_len,
        "seed": seed,
        "shift_bits": SHIFT_BITS,
        "wq_shape": list(wq_fp32.shape),
        "activation_shape": list(activation_fp32.shape),
    }
    return cases, header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MiniGPT checkpoint (.pt)",
    )
    parser.add_argument(
        "--learn-root",
        type=Path,
        default=DEFAULT_LEARN_ROOT,
        help="Root of Desktop/learn repo",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="Optional JSON metadata output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train with: cd learn/mini_gpt_ready && python models/train.py --mode train --steps 200"
        )

    cases, header = export_all_cases(
        args.checkpoint, args.learn_root, args.seq_len, args.seed
    )
    print(f"Exported {len(cases)} Wq slice cases from {args.checkpoint}")
    for case in cases:
        print(
            f"  case {case['case_id']}: M={case['m']} N={case['n']} K={case['k']} "
            f"meta={case['meta']['m_row_start']},{case['meta']['n_out_start']},"
            f"{case['meta']['k_in_start']}"
        )

    if args.meta_out is not None:
        payload = {"header": header, "cases": cases}
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()

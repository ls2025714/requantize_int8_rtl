#!/usr/bin/env python3
"""D7: Export MiniGPT Q/K/V × multi-head banks for int8_linear_tiled.

Memory map (frozen, see NOTES.md §0.2):
  region(op, head) = op_base[op] + head * 1024
  op_base = {WQ:0, WK:4096, WV:8192}; WEIGHT_DEPTH=12288
  mult_base(op, head) = op * 64 + head * 16; MULT_DEPTH=192
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from export_mini_gpt_head0_q_full import weight_tile_major_flat, verify_tile_major_matches_column_major
from export_mini_gpt_wq_slice import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LEARN_ROOT,
    MULT_MAX,
    MULT_MIN,
    SHIFT_BITS,
    activation_rtl_flat,
    expected_rtl_flat,
)

FULL_N = 16
FULL_K = 64
NUM_OPS = 3
NUM_HEADS = 4
HEAD_WEIGHT_ELEMS = FULL_N * FULL_K
OP_WEIGHT_ELEMS = NUM_HEADS * HEAD_WEIGHT_ELEMS
WEIGHT_DEPTH = NUM_OPS * OP_WEIGHT_ELEMS
MULT_DEPTH = NUM_OPS * NUM_HEADS * FULL_N
OP_BASE = (0, 4096, 8192)
OP_NAMES = ("WQ", "WK", "WV")


@dataclass(frozen=True)
class QkvCaseSpec:
    case_id: int
    op_type: int
    head_idx: int
    m: int
    m_row_start: int = 0
    a_gap: int = 0
    c_hold: int = 2


def _setup_import_paths(learn_root: Path) -> None:
    mini_gpt_models = learn_root / "mini_gpt_ready" / "models"
    ref_dir = learn_root / "int8_gemm_reference"
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


def region_base(op_type: int, head_idx: int) -> int:
    return OP_BASE[op_type] + head_idx * HEAD_WEIGHT_ELEMS


def mult_base(op_type: int, head_idx: int) -> int:
    return op_type * (NUM_HEADS * FULL_N) + head_idx * FULL_N


def capture_block0_norm1_and_qkv(
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

    captured: dict[str, Any] = {}

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
    qkv = qkv_weight.astype(np.float32)
    if qkv.shape != (192, 64):
        raise ValueError(f"expected qkv_proj [192,64], got {qkv.shape}")
    return activation, qkv


def extract_head_weight(qkv_fp32: np.ndarray, op_type: int, head_idx: int) -> np.ndarray:
    row0 = op_type * 64 + head_idx * FULL_N
    return qkv_fp32[row0 : row0 + FULL_N, :].astype(np.float32)


def build_case_specs() -> list[QkvCaseSpec]:
    return [
        QkvCaseSpec(case_id=1, op_type=0, head_idx=0, m=4),
        QkvCaseSpec(case_id=2, op_type=1, head_idx=0, m=4),
        QkvCaseSpec(case_id=3, op_type=2, head_idx=0, m=4),
        QkvCaseSpec(case_id=4, op_type=0, head_idx=1, m=4),
        QkvCaseSpec(case_id=5, op_type=2, head_idx=3, m=2),
    ]


def build_banks_and_cases(
    activation_fp32: np.ndarray,
    qkv_fp32: np.ndarray,
    specs: list[QkvCaseSpec],
    ref_fns,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    (
        quantize_per_tensor,
        quantize_weight_per_output_channel,
        int8_gemm_int32,
        requantize_accumulator_fixed,
        symmetric_scale,
    ) = ref_fns

    # Freeze quantization on M=4 window starting at row 0 (shared by all cases).
    a_window = activation_fp32[0:4, :]
    input_q_full, input_scale, _ = quantize_per_tensor(a_window)

    weight_bank = [0] * WEIGHT_DEPTH
    mult_bank = [0] * MULT_DEPTH
    head_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for op in range(NUM_OPS):
        for head in range(NUM_HEADS):
            w_fp32 = extract_head_weight(qkv_fp32, op, head)
            weight_q, weight_scales, _ = quantize_weight_per_output_channel(w_fp32)
            accumulator = int8_gemm_int32(input_q_full, weight_q)
            baseline_fp32 = a_window @ w_fp32.T
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
                        f"op={op} head={head}: multiplier {mult} out of 18-bit range"
                    )
            tile_weights = weight_tile_major_flat(weight_q)
            verify_tile_major_matches_column_major(weight_q, tile_weights)

            w_base = region_base(op, head)
            for i, val in enumerate(tile_weights):
                weight_bank[w_base + i] = val
            m_base = mult_base(op, head)
            for i, val in enumerate(mults):
                mult_bank[m_base + i] = val

            head_cache[(op, head)] = {
                "output_q": output_q,
                "mults": mults,
                "weight_scales": [float(v) for v in weight_scales.tolist()],
                "output_scale": float(output_scale),
                "base_addr": w_base,
                "mult_base": m_base,
            }

    cases: list[dict[str, Any]] = []
    for spec in specs:
        cached = head_cache[(spec.op_type, spec.head_idx)]
        m_end = spec.m_row_start + spec.m
        if m_end > 4:
            raise ValueError(f"case {spec.case_id}: m slice exceeds frozen 4-row window")
        a_q = input_q_full[spec.m_row_start:m_end, :]
        out_q = cached["output_q"][spec.m_row_start:m_end, :]
        cases.append(
            {
                "case_id": spec.case_id,
                "op_type": spec.op_type,
                "head_idx": spec.head_idx,
                "op_name": OP_NAMES[spec.op_type],
                "m": spec.m,
                "n": FULL_N,
                "k": FULL_K,
                "a_gap": spec.a_gap,
                "c_hold": spec.c_hold,
                "base_addr": cached["base_addr"],
                "mult_base": cached["mult_base"],
                "a": activation_rtl_flat(a_q),
                "expected": expected_rtl_flat(out_q),
                "meta": {
                    "m_row_start": spec.m_row_start,
                    "input_scale": float(input_scale),
                    "weight_scales": cached["weight_scales"],
                    "output_scale": cached["output_scale"],
                },
            }
        )

    banks_meta = {
        "weight_depth": WEIGHT_DEPTH,
        "mult_depth": MULT_DEPTH,
        "op_base": list(OP_BASE),
        "head_stride": HEAD_WEIGHT_ELEMS,
        "mult_stride_head": FULL_N,
        "input_scale": float(input_scale),
    }
    return weight_bank, mult_bank, cases, banks_meta


def export_all(
    checkpoint_path: Path,
    learn_root: Path,
    seq_len: int,
    seed: int,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    _setup_import_paths(learn_root)
    ref_fns = _load_reference()
    mini_gpt_models = learn_root / "mini_gpt_ready" / "models"
    activation_fp32, qkv_fp32 = capture_block0_norm1_and_qkv(
        checkpoint_path, mini_gpt_models, seq_len, seed
    )
    weight_bank, mult_bank, cases, banks_meta = build_banks_and_cases(
        activation_fp32, qkv_fp32, build_case_specs(), ref_fns
    )
    header = {
        "source": "mini_gpt Q/K/V multi-head tiled banks",
        "checkpoint": str(checkpoint_path.resolve()),
        "seq_len": seq_len,
        "seed": seed,
        "shift_bits": SHIFT_BITS,
        "qkv_shape": list(qkv_fp32.shape),
        "activation_shape": list(activation_fp32.shape),
        "banks": banks_meta,
    }
    return weight_bank, mult_bank, cases, header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learn-root", type=Path, default=DEFAULT_LEARN_ROOT)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--meta-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    weight_bank, mult_bank, cases, header = export_all(
        args.checkpoint, args.learn_root, args.seq_len, args.seed
    )
    print(f"Exported {len(cases)} QKV cases; WEIGHT={len(weight_bank)} MULT={len(mult_bank)}")
    for case in cases:
        print(
            f"  case {case['case_id']}: {case['op_name']} head={case['head_idx']} "
            f"M={case['m']} base={case['base_addr']} mult_base={case['mult_base']}"
        )
    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "header": header,
            "cases": cases,
            "weight_bank_len": len(weight_bank),
            "mult_bank_len": len(mult_bank),
        }
        args.meta_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()

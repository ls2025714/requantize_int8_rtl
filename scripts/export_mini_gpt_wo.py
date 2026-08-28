#!/usr/bin/env python3
"""D10: Export MiniGPT out_proj [64,64] for int8_linear_tiled Wo regression."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from export_mini_gpt_wq_slice import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LEARN_ROOT,
    MULT_MAX,
    MULT_MIN,
    SHIFT_BITS,
    activation_rtl_flat,
    expected_rtl_flat,
)

FULL_N = 64
FULL_K = 64
TILE_N = 4
TILE_K = 16
NUM_HEADS = 4
HEAD_DIM = 16


@dataclass(frozen=True)
class WoSpec:
    case_id: int
    m: int
    m_row_start: int = 0
    a_gap: int = 0
    c_hold: int = 2
    mode: str = "attn_concat"  # attn_concat | synth_heads


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


def weight_tile_major_flat(weight_q: np.ndarray) -> list[int]:
    weight_q = np.asarray(weight_q, dtype=np.int8)
    if weight_q.shape != (FULL_N, FULL_K):
        raise ValueError(f"expected Wo shape ({FULL_N}, {FULL_K}), got {weight_q.shape}")
    values: list[int] = []
    for n_base in range(0, FULL_N, TILE_N):
        for k_base in range(0, FULL_K, TILE_K):
            for kk in range(TILE_K):
                for j in range(TILE_N):
                    values.append(int(weight_q[n_base + j, k_base + kk]))
    if len(values) != FULL_N * FULL_K:
        raise ValueError(f"tile-major flat length {len(values)} != {FULL_N * FULL_K}")
    return values


def capture_attn_concat_and_wo(
    checkpoint_path: Path,
    mini_gpt_models: Path,
    seq_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (attn_concat_fp32 [T,64], wo_fp32 [64,64])."""
    import torch
    from mini_gpt import MiniGPT  # noqa: WPS433

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ckpt["model_config"]
    stoi = ckpt["stoi"]

    model = MiniGPT(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["attn_concat"] = inputs[0].detach().cpu()

    handle = model.blocks[0].attention.out_proj.register_forward_hook(hook)

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

    activation = captured["attn_concat"].numpy()[0].astype(np.float32)
    wo = ckpt["model_state_dict"]["blocks.0.attention.out_proj.weight"].numpy().astype(
        np.float32
    )
    if wo.shape != (FULL_N, FULL_K):
        raise ValueError(f"out_proj shape {wo.shape}, expected ({FULL_N},{FULL_K})")
    return activation, wo


def synth_multihead_concat(m: int, seed: int) -> np.ndarray:
    """Build [M,64] as concat of 4 head dims (token, head, head_dim)."""
    rng = np.random.default_rng(seed)
    heads = []
    for h in range(NUM_HEADS):
        heads.append(rng.normal(0.0, 1.0, size=(m, HEAD_DIM)).astype(np.float32))
    # layout: token, head, head_dim → flat [M,64]
    return np.concatenate(heads, axis=1)


def build_case_specs() -> list[WoSpec]:
    return [
        WoSpec(case_id=1, m=4, mode="attn_concat"),
        WoSpec(case_id=2, m=1, mode="attn_concat"),
        WoSpec(case_id=3, m=2, mode="attn_concat"),
        WoSpec(case_id=4, m=4, mode="synth_heads", a_gap=1, c_hold=3),
        WoSpec(case_id=5, m=4, m_row_start=0, mode="attn_concat", c_hold=2),
    ]


def export_case(
    spec: WoSpec,
    attn_fp32: np.ndarray,
    wo_fp32: np.ndarray,
    ref_fns,
    seed: int,
) -> dict[str, Any]:
    (
        quantize_per_tensor,
        quantize_weight_per_output_channel,
        int8_gemm_int32,
        requantize_accumulator_fixed,
        symmetric_scale,
    ) = ref_fns

    if spec.mode == "attn_concat":
        m_end = spec.m_row_start + spec.m
        if m_end > attn_fp32.shape[0]:
            raise ValueError(f"case {spec.case_id}: m slice exceeds activation rows")
        a_fp32 = attn_fp32[spec.m_row_start:m_end, :]
    else:
        a_fp32 = synth_multihead_concat(spec.m, seed + spec.case_id * 17)

    input_q, input_scale, _ = quantize_per_tensor(a_fp32)
    weight_q, weight_scales, _ = quantize_weight_per_output_channel(wo_fp32)
    accumulator = int8_gemm_int32(input_q, weight_q)

    baseline_fp32 = a_fp32 @ wo_fp32.T
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
        "n": FULL_N,
        "k": FULL_K,
        "a_gap": spec.a_gap,
        "c_hold": spec.c_hold,
        "mode": spec.mode,
        "a": activation_rtl_flat(input_q),
        "weights": weight_tile_major_flat(weight_q),
        "mults": mults,
        "expected": expected_rtl_flat(output_q),
        "meta": {
            "m_row_start": spec.m_row_start,
            "input_scale": float(input_scale),
            "output_scale": float(output_scale),
            "num_heads_concat": NUM_HEADS if spec.mode == "synth_heads" else None,
        },
    }


def export_all_cases(
    checkpoint_path: Path,
    learn_root: Path,
    seq_len: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _setup_import_paths(learn_root)
    ref_fns = _load_reference()
    mini_gpt_models = learn_root / "mini_gpt_ready" / "models"

    attn_fp32, wo_fp32 = capture_attn_concat_and_wo(
        checkpoint_path, mini_gpt_models, seq_len, seed
    )

    cases: list[dict[str, Any]] = []
    for spec in build_case_specs():
        cases.append(export_case(spec, attn_fp32, wo_fp32, ref_fns, seed))

    header = {
        "source": "mini_gpt out_proj Wo tiled N=64 K=64",
        "checkpoint": str(checkpoint_path.resolve()),
        "seq_len": seq_len,
        "seed": seed,
        "shift_bits": SHIFT_BITS,
        "wo_shape": [FULL_N, FULL_K],
        "activation_shape": list(attn_fp32.shape),
    }
    return cases, header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learn-root", type=Path, default=DEFAULT_LEARN_ROOT)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--meta-out", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    cases, header = export_all_cases(
        args.checkpoint, args.learn_root, args.seq_len, args.seed
    )
    print(f"Exported {len(cases)} Wo tiled cases from {args.checkpoint}")
    for case in cases:
        print(f"  case {case['case_id']}: M={case['m']} mode={case['mode']}")
    if args.meta_out is not None:
        args.meta_out.write_text(
            json.dumps({"header": header, "cases": cases}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

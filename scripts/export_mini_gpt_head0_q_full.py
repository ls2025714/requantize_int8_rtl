#!/usr/bin/env python3
"""D6: Export full head0 Wq [16,64] and norm1 activations for int8_linear_tiled."""

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
    capture_block0_norm1_activation,
    expected_rtl_flat,
)

FULL_N = 16
FULL_K = 64
TILE_N = 4
TILE_K = 16


@dataclass(frozen=True)
class FullTileSpec:
    case_id: int
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


def weight_tile_major_flat(weight_q: np.ndarray) -> list[int]:
    """Wq [N,K] -> 16 tile blocks × 64, each block K×N column-major."""
    weight_q = np.asarray(weight_q, dtype=np.int8)
    if weight_q.shape != (FULL_N, FULL_K):
        raise ValueError(f"expected Wq shape ({FULL_N}, {FULL_K}), got {weight_q.shape}")

    values: list[int] = []
    for n_base in range(0, FULL_N, TILE_N):
        for k_base in range(0, FULL_K, TILE_K):
            for kk in range(TILE_K):
                for j in range(TILE_N):
                    values.append(int(weight_q[n_base + j, k_base + kk]))
    if len(values) != FULL_N * FULL_K:
        raise ValueError(f"tile-major flat length {len(values)} != {FULL_N * FULL_K}")
    return values


def verify_tile_major_matches_column_major(weight_q: np.ndarray, tile_flat: list[int]) -> None:
    """Sanity: reconstruct [16,64] from tile blocks and compare."""
    weight_q = np.asarray(weight_q, dtype=np.int8)
    idx = 0
    reconstructed = np.zeros((FULL_N, FULL_K), dtype=np.int8)
    for n_base in range(0, FULL_N, TILE_N):
        for k_base in range(0, FULL_K, TILE_K):
            for kk in range(TILE_K):
                for j in range(TILE_N):
                    reconstructed[n_base + j, k_base + kk] = np.int8(tile_flat[idx])
                    idx += 1
    if not np.array_equal(reconstructed, weight_q):
        raise RuntimeError("tile-major reorder does not match original Wq")


def build_case_specs() -> list[FullTileSpec]:
    return [
        FullTileSpec(case_id=1, m=1),
        FullTileSpec(case_id=2, m=4),
        FullTileSpec(case_id=3, m=2),
        FullTileSpec(case_id=4, m=4, m_row_start=4),
    ]


def export_case(
    spec: FullTileSpec,
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
    if m_end > activation_fp32.shape[0]:
        raise ValueError(f"case {spec.case_id}: m slice exceeds activation rows")
    if wq_fp32.shape != (FULL_N, FULL_K):
        raise ValueError(f"case {spec.case_id}: Wq must be [{FULL_N},{FULL_K}]")

    a_fp32 = activation_fp32[spec.m_row_start:m_end, :]
    w_fp32 = wq_fp32

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

    tile_weights = weight_tile_major_flat(weight_q)
    verify_tile_major_matches_column_major(weight_q, tile_weights)

    return {
        "case_id": spec.case_id,
        "m": spec.m,
        "n": FULL_N,
        "k": FULL_K,
        "a_gap": spec.a_gap,
        "c_hold": spec.c_hold,
        "a": activation_rtl_flat(input_q),
        "weights": tile_weights,
        "mults": mults,
        "expected": expected_rtl_flat(output_q),
        "meta": {
            "m_row_start": spec.m_row_start,
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
    _setup_import_paths(learn_root)
    ref_fns = _load_reference()
    mini_gpt_models = learn_root / "mini_gpt_ready" / "models"

    activation_fp32, wq_all = capture_block0_norm1_activation(
        checkpoint_path, mini_gpt_models, seq_len, seed
    )
    wq_fp32 = wq_all[0:FULL_N, :].astype(np.float32)

    cases: list[dict[str, Any]] = []
    for spec in build_case_specs():
        cases.append(export_case(spec, activation_fp32, wq_fp32, ref_fns))

    header = {
        "source": "mini_gpt head0 Q full tiled N=16 K=64",
        "checkpoint": str(checkpoint_path.resolve()),
        "seq_len": seq_len,
        "seed": seed,
        "shift_bits": SHIFT_BITS,
        "wq_shape": [FULL_N, FULL_K],
        "activation_shape": list(activation_fp32.shape),
    }
    return cases, header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--learn-root", type=Path, default=DEFAULT_LEARN_ROOT)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
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
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cases, header = export_all_cases(
        args.checkpoint, args.learn_root, args.seq_len, args.seed
    )
    print(f"Exported {len(cases)} full-tile head0 Q cases from {args.checkpoint}")
    for case in cases:
        print(
            f"  case {case['case_id']}: M={case['m']} meta m_row_start="
            f"{case['meta']['m_row_start']}"
        )

    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(
            json.dumps({"header": header, "cases": cases}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote metadata: {args.meta_out}")


if __name__ == "__main__":
    main()

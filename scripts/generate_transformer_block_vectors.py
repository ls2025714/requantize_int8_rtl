#!/usr/bin/env python3
"""D12: Generate transformer_block_vectors.txt — one E2E MiniGPT block (seq=1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from export_mini_gpt_ffn import (
    D_FF,
    EMBD,
    capture_norm2_and_mlp,
    weight_tile_major_flat_nk,
)
from export_mini_gpt_qkv_heads import (
    FULL_K,
    FULL_N,
    MULT_DEPTH as QKV_MULT_DEPTH,
    NUM_HEADS,
    NUM_OPS,
    WEIGHT_DEPTH as QKV_WEIGHT_DEPTH,
    build_banks_and_cases,
    capture_block0_norm1_and_qkv,
    extract_head_weight,
    mult_base,
    region_base,
)
from export_mini_gpt_wq_slice import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LEARN_ROOT,
    SHIFT_BITS,
    activation_rtl_flat,
    expected_rtl_flat,
)
from export_mini_gpt_wo import FULL_K as WO_K, FULL_N as WO_N, weight_tile_major_flat
from silu_fixed_ref import silu_int8
from softmax_fixed_ref import fixed_softmax_matrix

SEED = 20260908
HEAD_DIM = 16
MAX_SEQ = 4
WO_WEIGHT_ELEMS = WO_N * WO_K
WO_MULT_ELEMS = WO_N
FFN_WIDE_ELEMS = D_FF * EMBD
FFN_WIDE_MULT = D_FF
FFN_DOWN_ELEMS = EMBD * D_FF
FFN_DOWN_MULT = EMBD


def _setup(learn_root: Path) -> None:
    for p in (learn_root / "mini_gpt_ready" / "models", learn_root / "int8_gemm_reference"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _ref():
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


def sat_add(a: int, b: int) -> int:
    s = int(a) + int(b)
    if s > 127:
        return 127
    if s < -127:
        return -127
    return s


def golden_requantize(acc: int, multiplier: int, shift_bits: int = SHIFT_BITS) -> int:
    acc_s = acc & 0xFFFFFFFF
    if acc_s >= 0x80000000:
        acc_s -= 0x100000000
    mult_u = multiplier & 0x3FFFF
    product = acc_s * mult_u
    half_lsb = 1 << (shift_bits - 1)
    if product >= 0:
        rounded = product + half_lsb
    else:
        rounded = product + (half_lsb - 1)
    shifted = rounded >> shift_bits
    if shifted > 127:
        return 127
    if shifted < -127:
        return -127
    return shifted


def unpack_tile_major(flat: list[int], full_n: int, full_k: int, tile_n: int = 4, tile_k: int = 16) -> np.ndarray:
    w = np.zeros((full_n, full_k), dtype=np.int8)
    idx = 0
    for n_base in range(0, full_n, tile_n):
        for k_base in range(0, full_k, tile_k):
            for kk in range(tile_k):
                for j in range(tile_n):
                    w[n_base + j, k_base + kk] = np.int8(flat[idx])
                    idx += 1
    return w


def linear_int8_with_mults(
    a: np.ndarray, w: np.ndarray, mults: list[int] | np.ndarray
) -> np.ndarray:
    """A[M,K] @ W[N,K]^T then per-col requant with frozen mults (RTL path)."""
    a = np.asarray(a, dtype=np.int8)
    w = np.asarray(w, dtype=np.int8)
    m, k = a.shape
    n = w.shape[0]
    out = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            acc = int(np.dot(a[i, :].astype(np.int32), w[j, :].astype(np.int32)))
            out[i, j] = np.int8(golden_requantize(acc, int(mults[j])))
    return out


def score_matmul(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return (q.astype(np.int32) @ k.astype(np.int32).T)


def attn_v_ref(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    seq, head_dim = v.shape
    half = 1 << 14
    out = np.zeros((seq, head_dim), dtype=np.int8)
    for i in range(seq):
        for d in range(head_dim):
            acc = sum(int(p[i, j]) * int(v[j, d]) for j in range(seq))
            rounded = acc + half if acc >= 0 else acc + (half - 1)
            shifted = rounded >> 15
            if shifted > 127:
                shifted = 127
            elif shifted < -127:
                shifted = -127
            out[i, d] = np.int8(shifted)
    return out


def elem_mul_rtl(a: int, b: int) -> int:
    prod = int(a) * int(b)
    if prod >= 0:
        shifted = (prod + 64) >> 7
    else:
        shifted = (prod + 63) >> 7
    if shifted > 127:
        return 127
    if shifted < -127:
        return -127
    return shifted


def run_linear_fp32(a_fp32: np.ndarray, w_fp32: np.ndarray, ref_fns):
    qpt, qwpc, gemm, requant, sym_scale = ref_fns
    iq, iscale, _ = qpt(a_fp32)
    wq, wscales, _ = qwpc(w_fp32)
    acc = gemm(iq, wq)
    base = a_fp32 @ w_fp32.T
    oscale = sym_scale(float(np.max(np.abs(base))))
    oq, _, _, mults, _ = requant(acc, iscale, wscales, oscale, SHIFT_BITS)
    return oq, mults, wq


def run_linear_int8(input_q: np.ndarray, w_fp32: np.ndarray, ref_fns):
    _, qwpc, gemm, requant, sym_scale = ref_fns
    wq, wscales, _ = qwpc(w_fp32)
    acc = gemm(input_q, wq)
    in_scale = sym_scale(float(np.max(np.abs(input_q.astype(np.float32)))))
    base = input_q.astype(np.float32) @ w_fp32.T
    oscale = sym_scale(float(np.max(np.abs(base))))
    oq, _, _, mults, _ = requant(acc, in_scale, wscales, oscale, SHIFT_BITS)
    return oq, mults, wq


def build_qkv_banks_m1(activation_fp32: np.ndarray, qkv_fp32: np.ndarray, ref_fns):
    """Build QKV weight/mult banks quantized on M=1 norm1 window."""
    qpt, qwpc, gemm, requant, sym_scale = ref_fns
    a_window = activation_fp32[0:1, :]
    input_q_full, input_scale, _ = qpt(a_window)

    weight_bank = [0] * QKV_WEIGHT_DEPTH
    mult_bank = [0] * QKV_MULT_DEPTH

    from export_mini_gpt_head0_q_full import weight_tile_major_flat, verify_tile_major_matches_column_major

    for op in range(NUM_OPS):
        for head in range(NUM_HEADS):
            w_fp32 = extract_head_weight(qkv_fp32, op, head)
            wq, weight_scales, _ = qwpc(w_fp32)
            acc = gemm(input_q_full, wq)
            baseline = a_window @ w_fp32.T
            output_scale = sym_scale(float(np.max(np.abs(baseline))))
            _, _, _, integer_multipliers, _ = requant(
                acc, input_scale, weight_scales, output_scale, SHIFT_BITS
            )
            tile_weights = weight_tile_major_flat(wq)
            verify_tile_major_matches_column_major(wq, tile_weights)
            w_base = region_base(op, head)
            for i, val in enumerate(tile_weights):
                weight_bank[w_base + i] = val
            m_base = mult_base(op, head)
            for i, val in enumerate(integer_multipliers.tolist()):
                mult_bank[m_base + i] = int(val)

    return weight_bank, mult_bank, input_q_full


def golden_e2e(
    checkpoint: Path,
    learn_root: Path,
    seed: int,
) -> dict:
    _setup(learn_root)
    ref = _ref()
    models = learn_root / "mini_gpt_ready" / "models"

    activation_fp32, qkv_fp32 = capture_block0_norm1_and_qkv(
        checkpoint, models, seq_len=1, seed=seed
    )
    _, gate_w, up_w, down_w = capture_norm2_and_mlp(checkpoint, models, seq_len=1, seed=seed)

    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    wo_fp32 = ckpt["model_state_dict"]["blocks.0.attention.out_proj.weight"].numpy().astype(
        np.float32
    )

    m = 1
    qkv_weight, qkv_mult, x_q = build_qkv_banks_m1(activation_fp32, qkv_fp32, ref)
    x_flat = x_q.reshape(-1)

    # Q/K/V with frozen banks (same mults RTL loads)
    q_heads, k_heads, v_heads = [], [], []
    for head in range(NUM_HEADS):
        for op, bucket in enumerate((q_heads, k_heads, v_heads)):
            w_flat = qkv_weight[region_base(op, head) : region_base(op, head) + FULL_N * FULL_K]
            w_mat = unpack_tile_major(w_flat, FULL_N, FULL_K)
            mults = qkv_mult[mult_base(op, head) : mult_base(op, head) + FULL_N]
            bucket.append(linear_int8_with_mults(x_q, w_mat, mults))

    attn_heads = []
    for head in range(NUM_HEADS):
        scores = score_matmul(q_heads[head], k_heads[head])
        probs = fixed_softmax_matrix(scores)
        attn_heads.append(attn_v_ref(probs, v_heads[head]))

    concat = np.concatenate(attn_heads, axis=1).astype(np.int8)

    # Wo bank: quantize once, reuse mults
    wo_out_tmp, wo_mults, wo_wq = run_linear_int8(concat, wo_fp32, ref)
    wo_weights = weight_tile_major_flat(wo_wq)
    wo_mat = unpack_tile_major(wo_weights, WO_N, WO_K)
    wo_out = linear_int8_with_mults(concat, wo_mat, wo_mults)

    res1 = np.zeros((m, EMBD), dtype=np.int8)
    for i in range(m):
        for j in range(EMBD):
            res1[i, j] = np.int8(sat_add(int(x_q[i, j]), int(wo_out[i, j])))

    # FFN banks: derive wq/mults once from res1, then apply frozen path
    gate_q_tmp, gate_mults, gate_wq = run_linear_int8(res1, gate_w, ref)
    up_q_tmp, up_mults, up_wq = run_linear_int8(res1, up_w, ref)
    gate_q = linear_int8_with_mults(res1, gate_wq, gate_mults)
    up_q = linear_int8_with_mults(res1, up_wq, up_mults)

    hidden = [
        elem_mul_rtl(silu_int8(int(g)), int(u))
        for g, u in zip(gate_q.reshape(-1), up_q.reshape(-1))
    ]
    hidden_q = np.array(hidden, dtype=np.int8).reshape(m, D_FF)

    ffn_tmp, down_mults, down_wq = run_linear_int8(hidden_q, down_w, ref)
    ffn_out = linear_int8_with_mults(hidden_q, down_wq, down_mults)

    res2 = np.zeros((m, EMBD), dtype=np.int8)
    for i in range(m):
        for j in range(EMBD):
            res2[i, j] = np.int8(sat_add(int(res1[i, j]), int(ffn_out[i, j])))

    return {
        "case_id": 1,
        "seq": m,
        "x": activation_rtl_flat(x_q),
        "qkv_weight": qkv_weight,
        "qkv_mult": qkv_mult,
        "wo_weight": wo_weights,
        "wo_mult": [int(v) for v in wo_mults.tolist()],
        "ffn_gate_weight": weight_tile_major_flat_nk(gate_wq, D_FF, EMBD),
        "ffn_gate_mult": [int(v) for v in gate_mults.tolist()],
        "ffn_up_weight": weight_tile_major_flat_nk(up_wq, D_FF, EMBD),
        "ffn_up_mult": [int(v) for v in up_mults.tolist()],
        "ffn_down_weight": weight_tile_major_flat_nk(down_wq, EMBD, D_FF),
        "ffn_down_mult": [int(v) for v in down_mults.tolist()],
        "expected": expected_rtl_flat(res2),
    }


def write_vectors(path: Path, seed: int, case: dict) -> None:
    lines = [
        f"# seed={seed} num_cases=1",
        "# source=mini_gpt block0 E2E seq=1 n_embd=64 n_head=4",
        "# PRELOAD_* once, then CASE id seq / X / EXPECT / END",
        "",
        "PRELOAD_QKV_WEIGHT",
    ]
    lines.extend(str(v) for v in case["qkv_weight"])
    lines.append("PRELOAD_QKV_MULT")
    lines.extend(str(v) for v in case["qkv_mult"])
    lines.append("PRELOAD_WO_WEIGHT")
    lines.extend(str(v) for v in case["wo_weight"])
    lines.append("PRELOAD_WO_MULT")
    lines.extend(str(v) for v in case["wo_mult"])
    lines.append("PRELOAD_FFN_GATE_WEIGHT")
    lines.extend(str(v) for v in case["ffn_gate_weight"])
    lines.append("PRELOAD_FFN_GATE_MULT")
    lines.extend(str(v) for v in case["ffn_gate_mult"])
    lines.append("PRELOAD_FFN_UP_WEIGHT")
    lines.extend(str(v) for v in case["ffn_up_weight"])
    lines.append("PRELOAD_FFN_UP_MULT")
    lines.extend(str(v) for v in case["ffn_up_mult"])
    lines.append("PRELOAD_FFN_DOWN_WEIGHT")
    lines.extend(str(v) for v in case["ffn_down_weight"])
    lines.append("PRELOAD_FFN_DOWN_MULT")
    lines.extend(str(v) for v in case["ffn_down_mult"])
    lines.append("")
    lines.append(f"CASE {case['case_id']} {case['seq']}")
    lines.append("X")
    lines.extend(str(v) for v in case["x"])
    lines.append("EXPECT")
    lines.extend(str(v) for v in case["expected"])
    lines.append("END")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--learn-root", type=Path, default=DEFAULT_LEARN_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "transformer_block_vectors.txt",
    )
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    case = golden_e2e(args.checkpoint, args.learn_root, args.seed)
    write_vectors(args.output, args.seed, case)
    print(f"Wrote 1 E2E case (seq={case['seq']}) to {args.output}")
    print(f"  seed={args.seed} X={len(case['x'])} EXPECT={len(case['expected'])}")


if __name__ == "__main__":
    main()

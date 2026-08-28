#!/usr/bin/env python3
"""D11: Export MiniGPT SwiGLU FFN weights + norm2 activation."""

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

EMBD = 64
D_FF = 256
TILE_N = 4
TILE_K = 16


@dataclass(frozen=True)
class FfnSpec:
    case_id: int
    m: int
    m_row_start: int = 0
    stage: str = "gate"  # gate | up | down | full


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


def weight_tile_major_flat_nk(weight_q: np.ndarray, full_n: int, full_k: int) -> list[int]:
    weight_q = np.asarray(weight_q, dtype=np.int8)
    if weight_q.shape != (full_n, full_k):
        raise ValueError(f"expected ({full_n},{full_k}), got {weight_q.shape}")
    values: list[int] = []
    for n_base in range(0, full_n, TILE_N):
        for k_base in range(0, full_k, TILE_K):
            for kk in range(TILE_K):
                for j in range(TILE_N):
                    values.append(int(weight_q[n_base + j, k_base + kk]))
    return values


def capture_norm2_and_mlp(checkpoint_path: Path, mini_gpt_models: Path, seq_len: int, seed: int):
    import torch
    from mini_gpt import MiniGPT  # noqa: WPS433

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MiniGPT(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, o):
        captured["norm2_out"] = o.detach().cpu()

    h = model.blocks[0].norm2.register_forward_hook(hook)
    rng = np.random.default_rng(seed)
    stoi = ckpt["stoi"]
    text_path = mini_gpt_models.parent / "data" / "input.txt"
    if text_path.exists():
        text = text_path.read_text(encoding="utf-8")
        start = int(rng.integers(0, max(1, len(text) - seq_len)))
        token_ids = [stoi[c] for c in text[start : start + seq_len] if c in stoi]
        while len(token_ids) < seq_len:
            token_ids.append(next(iter(stoi.values())))
        token_ids = token_ids[:seq_len]
    else:
        token_ids = rng.integers(0, ckpt["model_config"]["vocab_size"], size=seq_len).tolist()
    with torch.no_grad():
        model(torch.tensor([token_ids], dtype=torch.long))
    h.remove()
    sd = ckpt["model_state_dict"]
    act = captured["norm2_out"].numpy()[0].astype(np.float32)
    gate_w = sd["blocks.0.mlp.gate_proj.weight"].numpy().astype(np.float32)
    up_w = sd["blocks.0.mlp.up_proj.weight"].numpy().astype(np.float32)
    down_w = sd["blocks.0.mlp.down_proj.weight"].numpy().astype(np.float32)
    return act, gate_w, up_w, down_w


def export_down_case(spec: FfnSpec, act_fp32: np.ndarray, gate_w, up_w, down_w, ref_fns):
    from silu_fixed_ref import silu_int8

    qpt, qwpc, gemm, requant, sym_scale = ref_fns
    a_fp32 = act_fp32[spec.m_row_start : spec.m_row_start + spec.m, :EMBD]

    def run_linear_fp32(a_f, w_f):
        iq, iscale, _ = qpt(a_f)
        wq, wscales, _ = qwpc(w_f)
        acc = gemm(iq, wq)
        base = a_f @ w_f.T
        oscale = sym_scale(float(np.max(np.abs(base))))
        oq, _, _, mults, _ = requant(acc, iscale, wscales, oscale, SHIFT_BITS)
        return oq, mults, wq

    def run_linear_int8(input_q: np.ndarray, w_f):
        wq, wscales, _ = qwpc(w_f)
        acc = gemm(input_q, wq)
        in_scale = sym_scale(float(np.max(np.abs(input_q.astype(np.float32)))))
        base = input_q.astype(np.float32) @ w_f.T
        oscale = sym_scale(float(np.max(np.abs(base))))
        oq, _, _, mults, _ = requant(acc, in_scale, wscales, oscale, SHIFT_BITS)
        return oq, mults, wq

    gate_q, _, _ = run_linear_fp32(a_fp32, gate_w)
    up_q, _, _ = run_linear_fp32(a_fp32, up_w)
    gate_flat = gate_q.reshape(-1)
    up_flat = up_q.reshape(-1)
    hidden = []
    for g, u in zip(gate_flat.tolist(), up_flat.tolist()):
        sg = silu_int8(int(g))
        prod = int(sg) * int(u)
        if prod >= 0:
            shifted = (prod + 64) >> 7
        else:
            shifted = (prod + 63) >> 7
        if shifted > 127:
            shifted = 127
        elif shifted < -127:
            shifted = -127
        hidden.append(shifted)
    hidden_q = np.array(hidden, dtype=np.int8).reshape(spec.m, D_FF)
    output_q, mult_list, weight_q = run_linear_int8(hidden_q, down_w)
    w_flat = weight_tile_major_flat_nk(weight_q, EMBD, D_FF)
    return {
        "case_id": spec.case_id,
        "stage": spec.stage,
        "m": spec.m,
        "n": EMBD,
        "k": D_FF,
        "a": [int(v) for v in hidden_q.reshape(-1)],
        "weights": w_flat,
        "mults": [int(v) for v in mult_list.tolist()],
        "expected": expected_rtl_flat(output_q),
    }


def export_linear_case(
    spec: FfnSpec,
    act_fp32: np.ndarray,
    w_fp32: np.ndarray,
    full_n: int,
    full_k: int,
    ref_fns,
) -> dict[str, Any]:
    qpt, qwpc, gemm, requant, sym_scale = ref_fns
    a_fp32 = act_fp32[spec.m_row_start : spec.m_row_start + spec.m, :EMBD]
    w = w_fp32
    if w.shape != (full_n, full_k):
        raise ValueError(f"case {spec.case_id}: W shape {w.shape} != ({full_n},{full_k})")
    input_q, input_scale, _ = qpt(a_fp32)
    weight_q, weight_scales, _ = qwpc(w)
    acc = gemm(input_q, weight_q)
    baseline = a_fp32 @ w.T
    out_scale = sym_scale(float(np.max(np.abs(baseline))))
    output_q, _, _, mults, _ = requant(acc, input_scale, weight_scales, out_scale, SHIFT_BITS)
    mult_list = [int(v) for v in mults.tolist()]
    for m in mult_list:
        if m < MULT_MIN or m > MULT_MAX:
            raise ValueError(f"mult {m} OOR")
    if full_k == EMBD:
        w_flat = weight_tile_major_flat_nk(weight_q, full_n, full_k)
        reconstructed = np.zeros((full_n, full_k), dtype=np.int8)
        idx = 0
        for n_base in range(0, full_n, TILE_N):
            for k_base in range(0, full_k, TILE_K):
                for kk in range(TILE_K):
                    for j in range(TILE_N):
                        reconstructed[n_base + j, k_base + kk] = np.int8(w_flat[idx])
                        idx += 1
        if not np.array_equal(reconstructed, weight_q):
            raise RuntimeError(f"case {spec.case_id}: tile-major mismatch")
    else:
        w_flat = weight_tile_major_flat_nk(weight_q, full_n, full_k)
    return {
        "case_id": spec.case_id,
        "stage": spec.stage,
        "m": spec.m,
        "n": full_n,
        "k": full_k,
        "a": activation_rtl_flat(input_q),
        "weights": w_flat,
        "mults": mult_list,
        "expected": expected_rtl_flat(output_q),
    }


def build_specs() -> list[FfnSpec]:
    return [
        FfnSpec(1, 1, 0, "gate"),
        FfnSpec(2, 1, 0, "up"),
        FfnSpec(3, 1, 0, "down"),
    ]


def export_all(checkpoint: Path, learn_root: Path, seq_len: int, seed: int):
    _setup(learn_root)
    ref = _ref()
    models = learn_root / "mini_gpt_ready" / "models"
    act, gate_w, up_w, down_w = capture_norm2_and_mlp(checkpoint, models, seq_len, seed)
    cases = []
    for spec in build_specs():
        if spec.stage == "gate":
            cases.append(export_linear_case(spec, act, gate_w, D_FF, EMBD, ref))
        elif spec.stage == "up":
            cases.append(export_linear_case(spec, act, up_w, D_FF, EMBD, ref))
        else:
            cases.append(export_down_case(spec, act, gate_w, up_w, down_w, ref))
    header = {
        "source": "mini_gpt SwiGLU FFN layered gate/up/down",
        "checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "d_ff": D_FF,
        "embd": EMBD,
    }
    return cases, header


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--learn-root", type=Path, default=DEFAULT_LEARN_ROOT)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    cases, hdr = export_all(args.checkpoint, args.learn_root, 32, args.seed)
    print(f"Exported {len(cases)} FFN stage cases")

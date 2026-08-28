#!/usr/bin/env python3
"""D7: Generate tiled_qkv_vectors.txt for tb_linear_tiled_qkv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_mini_gpt_qkv_heads import (
    FULL_K,
    FULL_N,
    MULT_DEPTH,
    WEIGHT_DEPTH,
    export_all,
    mult_base,
    region_base,
)
from export_mini_gpt_wq_slice import DEFAULT_CHECKPOINT, SHIFT_BITS

D7_SEED = 20260831


def write_vector_file(
    path: Path,
    seed: int,
    weight_bank: list[int],
    mult_bank: list[int],
    cases: list[dict],
    source: str,
) -> None:
    lines: list[str] = [
        f"# seed={seed} num_cases={len(cases)}",
        f"# source={source}",
        f"# WEIGHT_DEPTH={WEIGHT_DEPTH} MULT_DEPTH={MULT_DEPTH}",
        "# memory_map: op_base={{0,4096,8192}} head_stride=1024 "
        "mult_base=op*64+head*16",
        "# format: PRELOAD_WEIGHT / PRELOAD_MULT once, then",
        "#         CASE id op head M a_gap c_hold",
        "#         A / EXPECT / END",
        "",
        "PRELOAD_WEIGHT",
    ]
    lines.extend(str(v) for v in weight_bank)
    lines.append("PRELOAD_MULT")
    lines.extend(str(v) for v in mult_bank)
    lines.append("")

    for case in cases:
        lines.append(
            "CASE {case_id} {op_type} {head_idx} {m} {a_gap} {c_hold}".format(**case)
        )
        lines.append("A")
        lines.extend(str(v) for v in case["a"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in case["expected"])
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def self_check(
    weight_bank: list[int],
    mult_bank: list[int],
    cases: list[dict],
) -> None:
    def to_s32(value: int) -> int:
        value &= 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
        return value

    def golden_requantize(acc: int, multiplier: int) -> int:
        acc_s = to_s32(acc)
        mult_u = multiplier & 0x3FFFF
        product = acc_s * mult_u
        half_lsb = 1 << (SHIFT_BITS - 1)
        if product >= 0:
            rounded = product + half_lsb
        else:
            rounded = product + (half_lsb - 1)
        shifted = rounded >> SHIFT_BITS
        if shifted > 127:
            return 127
        if shifted < -127:
            return -127
        return shifted

    tile_n = 4
    tile_k = 16
    for case in cases:
        op = int(case["op_type"])
        head = int(case["head_idx"])
        m = int(case["m"])
        a = case["a"]
        expected = case["expected"]
        w_base = region_base(op, head)
        m_base = mult_base(op, head)
        assert case["base_addr"] == w_base
        assert case["mult_base"] == m_base

        w_mat = [[0] * FULL_K for _ in range(FULL_N)]
        idx = w_base
        for n_base in range(0, FULL_N, tile_n):
            for k_base in range(0, FULL_K, tile_k):
                for kk in range(tile_k):
                    for j in range(tile_n):
                        w_mat[n_base + j][k_base + kk] = int(weight_bank[idx])
                        idx += 1

        for i in range(m):
            for j in range(FULL_N):
                acc = 0
                for kk in range(FULL_K):
                    acc += int(a[i * FULL_K + kk]) * w_mat[j][kk]
                got = golden_requantize(acc, int(mult_bank[m_base + j]))
                exp = int(expected[i * FULL_N + j])
                if got != exp:
                    raise RuntimeError(
                        f"self-check failed case={case['case_id']} "
                        f"op={op} head={head} C[{i}][{j}] got={got} expected={exp}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=D7_SEED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--learn-root",
        type=Path,
        default=Path(r"F:/Users/22563/Desktop/learn"),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tiled_qkv_vectors.txt",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tiled_qkv_export_meta.json",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt.exists():
        ckpt = DEFAULT_CHECKPOINT

    weight_bank, mult_bank, cases, header = export_all(
        ckpt, args.learn_root, args.seq_len, args.seed
    )
    self_check(weight_bank, mult_bank, cases)
    write_vector_file(
        args.output, args.seed, weight_bank, mult_bank, cases, header["source"]
    )
    args.meta_out.write_text(
        json.dumps(
            {
                "header": header,
                "cases": [
                    {k: v for k, v in c.items() if k not in ("a", "expected")}
                    for c in cases
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"  seed={args.seed} WEIGHT={len(weight_bank)} MULT={len(mult_bank)}")
    print(f"  metadata: {args.meta_out}")
    print("Self-check: PASS")


if __name__ == "__main__":
    main()

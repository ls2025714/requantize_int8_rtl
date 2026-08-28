#!/usr/bin/env python3
"""D10: Generate tiled_wo_vectors.txt for tb_linear_tiled_wo."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_mini_gpt_wq_slice import DEFAULT_CHECKPOINT, SHIFT_BITS
from export_mini_gpt_wo import FULL_K, FULL_N, export_all_cases

D10_WO_SEED = 20260904


def write_vector_file(path: Path, seed: int, cases: list[dict], source: str) -> None:
    lines: list[str] = [
        f"# seed={seed} num_cases={len(cases)}",
        f"# source={source}",
        f"# FULL_N={FULL_N} FULL_K={FULL_K} NUM_OPS=1 NUM_HEADS=1",
        "# format: CASE id M a_gap c_hold",
        "#         A / WEIGHT / MULT / EXPECT / END",
        "",
    ]
    for case in cases:
        lines.append(
            "CASE {case_id} {m} {a_gap} {c_hold}".format(**case)
        )
        lines.append("A")
        lines.extend(str(v) for v in case["a"])
        lines.append("WEIGHT")
        lines.extend(str(v) for v in case["weights"])
        lines.append("MULT")
        lines.extend(str(v) for v in case["mults"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in case["expected"])
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def self_check_cases(cases: list[dict]) -> None:
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

    for case in cases:
        m, n, k = case["m"], case["n"], case["k"]
        a = case["a"]
        w = case["weights"]
        mults = case["mults"]
        expected = case["expected"]
        tile_n, tile_k = 4, 16
        w_mat = [[0] * k for _ in range(n)]
        idx = 0
        for n_base in range(0, n, tile_n):
            for k_base in range(0, k, tile_k):
                for kk in range(tile_k):
                    for j in range(tile_n):
                        w_mat[n_base + j][k_base + kk] = int(w[idx])
                        idx += 1
        for i in range(m):
            for j in range(n):
                acc = 0
                for kk in range(k):
                    acc += int(a[i * k + kk]) * w_mat[j][kk]
                got = golden_requantize(acc, mults[j])
                exp = int(expected[i * n + j])
                if got != exp:
                    raise RuntimeError(
                        f"self-check failed case={case['case_id']} C[{i}][{j}] "
                        f"got={got} expected={exp}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=D10_WO_SEED)
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
        default=Path(__file__).resolve().parent.parent / "tiled_wo_vectors.txt",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tiled_wo_export_meta.json",
    )
    args = parser.parse_args()
    ckpt = args.checkpoint
    if not ckpt.exists():
        ckpt = DEFAULT_CHECKPOINT

    cases, header = export_all_cases(ckpt, args.learn_root, args.seq_len, args.seed)
    self_check_cases(cases)
    write_vector_file(args.output, args.seed, cases, header["source"])
    args.meta_out.write_text(
        __import__("json").dumps(
            {"header": header, "cases": cases}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"  seed={args.seed} source={header['source']}")
    print("Self-check: PASS")


if __name__ == "__main__":
    main()

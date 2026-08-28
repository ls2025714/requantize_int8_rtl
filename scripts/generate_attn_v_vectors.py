#!/usr/bin/env python3
"""D10: Generate attn_v_vectors.txt — P(UQ1.15) @ V(INT8) → INT8."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softmax_fixed_ref import ONE_Q15, fixed_softmax_matrix

SEED = 20260903
HEAD_DIM = 16
SHIFT = 15
HALF = 1 << (SHIFT - 1)


def attn_v(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    seq = p.shape[0]
    out = np.zeros((seq, HEAD_DIM), dtype=np.int8)
    for i in range(seq):
        for d in range(HEAD_DIM):
            acc = 0
            for j in range(seq):
                acc += int(p[i, j]) * int(v[j, d])
            if acc >= 0:
                rounded = acc + HALF
            else:
                rounded = acc + (HALF - 1)
            shifted = rounded >> SHIFT
            if shifted > 127:
                shifted = 127
            elif shifted < -127:
                shifted = -127
            out[i, d] = np.int8(shifted)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "attn_v_vectors.txt",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cases = []
    # deterministic
    scores = np.array([[10, 0, 0, 0], [5, 5, 0, 0], [1, 1, 1, 0], [2, 2, 2, 2]], dtype=np.int32)
    p = fixed_softmax_matrix(scores)
    v = np.arange(4 * HEAD_DIM, dtype=np.int8).reshape(4, HEAD_DIM) - 64
    cases.append({"case_id": 1, "seq": 4, "hold": 1, "p": p, "v": v, "expected": attn_v(p, v)})

    for i in range(4):
        seq = int(rng.integers(2, 5))
        s = rng.integers(-1000, 1000, size=(seq, seq), dtype=np.int32)
        p = fixed_softmax_matrix(s)
        v = rng.integers(-128, 128, size=(seq, HEAD_DIM), dtype=np.int8)
        cases.append(
            {"case_id": 2 + i, "seq": seq, "hold": int(rng.integers(0, 2)), "p": p, "v": v, "expected": attn_v(p, v)}
        )

    lines = [f"# seed={args.seed} num_cases={len(cases)}", ""]
    for c in cases:
        seq = c["seq"]
        lines.append(f"CASE {c['case_id']} {seq} {c['hold']}")
        lines.append("P")
        lines.extend(str(int(x)) for x in c["p"].reshape(-1))
        lines.append("V")
        lines.extend(str(int(x)) for x in c["v"].reshape(-1))
        lines.append("EXPECT")
        lines.extend(str(int(x)) for x in c["expected"].reshape(-1))
        lines.append("END")
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()

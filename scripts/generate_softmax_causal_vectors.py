#!/usr/bin/env python3
"""D9: Generate softmax_causal_vectors.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softmax_fixed_ref import fixed_softmax_matrix

SEED = 20260902


def build_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    # Case 1: seq=2 simple
    s1 = np.array([[10, 100], [20, 30]], dtype=np.int32)
    cases.append({"case_id": 1, "seq": 2, "hold": 1, "score": s1, "expected": fixed_softmax_matrix(s1)})
    # Case 2: seq=4 increasing
    s2 = np.arange(16, dtype=np.int32).reshape(4, 4) * 10
    cases.append({"case_id": 2, "seq": 4, "hold": 0, "score": s2, "expected": fixed_softmax_matrix(s2)})
    # Case 3: extremes
    s3 = np.array([[1000000, -1000000, 0, 5], [1, 2, 3, 4], [-50, -40, -30, -20], [7, 7, 7, 7]], dtype=np.int32)
    cases.append({"case_id": 3, "seq": 4, "hold": 2, "score": s3, "expected": fixed_softmax_matrix(s3)})
    for i in range(3):
        seq = int(rng.integers(2, 5))
        s = rng.integers(-50000, 50000, size=(seq, seq), dtype=np.int32)
        cases.append(
            {
                "case_id": 4 + i,
                "seq": seq,
                "hold": int(rng.integers(0, 3)),
                "score": s,
                "expected": fixed_softmax_matrix(s),
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "softmax_causal_vectors.txt",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng)
    lines = [f"# seed={args.seed} num_cases={len(cases)}", ""]
    for c in cases:
        seq = c["seq"]
        lines.append(f"CASE {c['case_id']} {seq} {c['hold']}")
        lines.append("SCORE")
        for v in c["score"].reshape(-1):
            lines.append(str(int(v)))
        lines.append("EXPECT")
        for v in c["expected"].reshape(-1):
            lines.append(str(int(v)))
        lines.append("END")
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()

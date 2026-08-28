#!/usr/bin/env python3
"""D8: Generate score_gemm_vectors.txt for Q @ K^T INT32 regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEED = 20260901
HEAD_DIM = 16
MAX_SEQ = 4


def score_matmul(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Score = Q @ K^T, all int."""
    return (q.astype(np.int32) @ k.astype(np.int32).T)


def build_cases(rng: np.random.Generator) -> list[dict]:
    cases: list[dict] = []

    # Case 1: seq=2 hand-check friendly
    q1 = np.array(
        [
            [1, 2, 3, 4] + [0] * 12,
            [5, 6, 7, 8] + [0] * 12,
        ],
        dtype=np.int8,
    )
    k1 = np.array(
        [
            [1, 0, 0, 0] + [0] * 12,
            [0, 1, 0, 0] + [0] * 12,
        ],
        dtype=np.int8,
    )
    cases.append(
        {
            "case_id": 1,
            "head": 0,
            "seq": 2,
            "q_gap": 0,
            "k_gap": 0,
            "s_hold": 1,
            "q": q1.reshape(-1).tolist(),
            "k": k1.reshape(-1).tolist(),
            "expected": score_matmul(q1, k1).reshape(-1).tolist(),
        }
    )

    # Case 2: seq=4 all ones / identity-like
    q2 = np.ones((4, HEAD_DIM), dtype=np.int8)
    k2 = np.eye(4, HEAD_DIM, dtype=np.int8)
    cases.append(
        {
            "case_id": 2,
            "head": 0,
            "seq": 4,
            "q_gap": 0,
            "k_gap": 0,
            "s_hold": 2,
            "q": q2.reshape(-1).tolist(),
            "k": k2.reshape(-1).tolist(),
            "expected": score_matmul(q2, k2).reshape(-1).tolist(),
        }
    )

    # Case 3: signed extremes
    q3 = np.full((4, HEAD_DIM), 127, dtype=np.int8)
    k3 = np.full((4, HEAD_DIM), -128, dtype=np.int8)
    cases.append(
        {
            "case_id": 3,
            "head": 1,
            "seq": 4,
            "q_gap": 1,
            "k_gap": 0,
            "s_hold": 1,
            "q": q3.reshape(-1).tolist(),
            "k": k3.reshape(-1).tolist(),
            "expected": score_matmul(q3, k3).reshape(-1).tolist(),
        }
    )

    # Cases 4-8: random multi-head-labeled
    for idx in range(5):
        seq = int(rng.integers(2, MAX_SEQ + 1))
        q = rng.integers(-128, 128, size=(seq, HEAD_DIM), dtype=np.int8)
        k = rng.integers(-128, 128, size=(seq, HEAD_DIM), dtype=np.int8)
        cases.append(
            {
                "case_id": 4 + idx,
                "head": int(idx % 4),
                "seq": seq,
                "q_gap": int(rng.integers(0, 3)),
                "k_gap": int(rng.integers(0, 3)),
                "s_hold": int(rng.integers(0, 3)),
                "q": q.reshape(-1).tolist(),
                "k": k.reshape(-1).tolist(),
                "expected": score_matmul(q, k).reshape(-1).tolist(),
            }
        )
    return cases


def write_vectors(path: Path, seed: int, cases: list[dict]) -> None:
    lines = [
        f"# seed={seed} num_cases={len(cases)}",
        "# format: CASE id head seq q_gap k_gap s_hold",
        "#         Q / K / EXPECT(INT32 row-major seq*seq) / END",
        "",
    ]
    for c in cases:
        lines.append(
            "CASE {case_id} {head} {seq} {q_gap} {k_gap} {s_hold}".format(**c)
        )
        lines.append("Q")
        lines.extend(str(v) for v in c["q"])
        lines.append("K")
        lines.extend(str(v) for v in c["k"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in c["expected"])
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "score_gemm_vectors.txt",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = build_cases(rng)
    write_vectors(args.output, args.seed, cases)
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()

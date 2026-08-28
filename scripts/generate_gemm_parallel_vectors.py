#!/usr/bin/env python3
"""D3：生成并行 GEMM 测试向量（INT32 期望）→ gemm_parallel_vectors.txt"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def golden_gemm(m: int, n: int, k: int, a: list[int], b: list[int]) -> list[int]:
    c = [0] * (m * n)
    for i in range(m):
        for j in range(n):
            total = 0
            for kk in range(k):
                total += int(a[i * k + kk]) * int(b[kk * n + j])
            c[i * n + j] = total
    return c


def make_case(
    case_id: int,
    m: int,
    n: int,
    k: int,
    a: list[int],
    b: list[int],
    a_gap: int,
    b_gap: int,
    c_hold: int,
) -> dict:
    if len(a) < m * k or len(b) < k * n:
        raise ValueError(f"case {case_id}: matrix size mismatch")
    expected = golden_gemm(m, n, k, a, b)
    return {
        "case_id": case_id,
        "m": m,
        "n": n,
        "k": k,
        "a_gap": a_gap,
        "b_gap": b_gap,
        "c_hold": c_hold,
        "a": a[: m * k],
        "b": b[: k * n],
        "expected": expected,
    }


def directed_cases() -> list[dict]:
    cases: list[dict] = []

    def add(
        case_id: int,
        m: int,
        n: int,
        k: int,
        a: list[int],
        b: list[int],
        a_gap: int = 0,
        b_gap: int = 0,
        c_hold: int = 2,
    ) -> None:
        cases.append(make_case(case_id, m, n, k, a, b, a_gap, b_gap, c_hold))

    add(1, 2, 2, 3, [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12])
    add(2, 1, 1, 1, [88], [91])
    add(3, 2, 3, 4, [i - 8 for i in range(8)], [((i * 5) % 255) - 128 for i in range(12)])
    add(4, 4, 4, 16, [((i * 3) % 255) - 128 for i in range(64)], [((i * 7) % 255) - 128 for i in range(64)], c_hold=3)

    return cases


def random_cases(rng: random.Random, start_id: int, count: int) -> list[dict]:
    cases: list[dict] = []
    for offset in range(count):
        case_id = start_id + offset
        m = rng.randint(1, 4)
        n = rng.randint(1, 4)
        k = rng.randint(1, 16)
        a = [rng.randint(-128, 127) for _ in range(m * k)]
        b = [rng.randint(-128, 127) for _ in range(k * n)]
        a_gap = rng.randint(0, 2)
        b_gap = rng.randint(0, 2)
        c_hold = rng.randint(0, 3)
        cases.append(make_case(case_id, m, n, k, a, b, a_gap, b_gap, c_hold))
    return cases


def write_vector_file(path: Path, seed: int, cases: list[dict]) -> None:
    lines: list[str] = [
        f"# seed={seed} num_cases={len(cases)}",
        "# format: CASE id M N K a_gap b_gap c_hold",
        "#         A",
        "#         <M*K signed INT8 values, one per line>",
        "#         B",
        "#         <K*N signed INT8 values, one per line>",
        "#         EXPECT",
        "#         <M*N signed INT32 values, one per line>",
        "#         END",
        "",
    ]
    for case in cases:
        lines.append(
            "CASE {case_id} {m} {n} {k} {a_gap} {b_gap} {c_hold}".format(**case)
        )
        lines.append("A")
        lines.extend(str(v) for v in case["a"])
        lines.append("B")
        lines.extend(str(v) for v in case["b"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in case["expected"])
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260827, help="random seed")
    parser.add_argument("--random-count", type=int, default=20, help="number of random cases")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "gemm_parallel_vectors.txt",
        help="output vector file path",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    directed = directed_cases()
    random_start = max(c["case_id"] for c in directed) + 1 if directed else 100
    random_part = random_cases(rng, random_start, args.random_count)
    cases = directed + random_part

    write_vector_file(args.output, args.seed, cases)
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"  directed={len(directed)} random={len(random_part)} seed={args.seed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task D：生成 4-lane 点积 FSM 测试向量 → dot_product_parallel_vectors.txt"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def beat_count(k: int) -> int:
    return (k + 3) // 4


def beat_keep_mask(k: int, beat_idx: int) -> int:
    base = beat_idx * 4
    mask = 0
    for lane in range(4):
        if base + lane < k:
            mask |= 1 << lane
    return mask


def golden_dot_product(k: int, a: list[int], b: list[int]) -> int:
    total = 0
    for i in range(k):
        total += int(a[i]) * int(b[i])
    return total


def pack_beats(k: int, a: list[int], b: list[int]) -> list[tuple[list[int], list[int], int]]:
    beats: list[tuple[list[int], list[int], int]] = []
    for beat in range(beat_count(k)):
        base = beat * 4
        av = [a[base + lane] if base + lane < k else 0 for lane in range(4)]
        bv = [b[base + lane] if base + lane < k else 0 for lane in range(4)]
        keep = beat_keep_mask(k, beat)
        beats.append((av, bv, keep))
    return beats


def make_case(
    case_id: int,
    k: int,
    a: list[int],
    b: list[int],
    gap: int,
    hold: int,
) -> dict:
    if len(a) < k or len(b) < k:
        raise ValueError(f"case {case_id}: vector length < K={k}")
    expected = golden_dot_product(k, a, b)
    return {
        "case_id": case_id,
        "k": k,
        "gap": gap,
        "hold": hold,
        "expected": expected,
        "beats": pack_beats(k, a, b),
    }


def directed_cases() -> list[dict]:
    cases: list[dict] = []

    def add(case_id: int, k: int, a: list[int], b: list[int], gap: int = 0, hold: int = 2) -> None:
        cases.append(make_case(case_id, k, a, b, gap, hold))

    add(1, 1, [7], [5])
    add(2, 3, [2, -4, 6], [3, 5, -1])
    add(3, 4, [1, 3, -5, 7], [2, 4, 6, -8])
    add(4, 5, [1, 2, 3, 4, 5], [2, -3, 2, -3, 2])
    add(5, 7, [10, 11, 12, 13, 14, 15, 16], [1, 1, 1, 1, 1, 1, 1])
    add(6, 8, [3, -2, 3, -2, 3, -2, 3, -2], [1, 2, 3, 4, 5, 6, 7, 8])
    add(7, 16, [i - 8 for i in range(16)], [4 if i % 3 == 0 else -1 for i in range(16)])
    add(8, 17, [17 - i for i in range(17)], [2] * 17)
    add(9, 4, [-128, 127, -1, 0], [-128, 127, 1, 0])
    add(10, 7, [10, 11, 12, 13, 14, 15, 16], [1, 1, 1, 1, 1, 1, 1], gap=2)
    add(11, 1, [-128], [-128])
    add(12, 2, [127, -128], [-128, 127])
    add(13, 4, [0, 0, 0, 0], [0, 0, 0, 0])
    add(14, 256, [1 if i % 2 == 0 else -1 for i in range(256)], [1] * 256, hold=3)
    add(15, 64, [((i * 7) % 255) - 128 for i in range(64)], [((i * 13) % 255) - 128 for i in range(64)], gap=3)

    # Back-to-back style: two short cases with minimal spacing metadata (same gap/hold per case).
    add(16, 4, [2, 3, 4, 5], [3, 3, 3, 3], hold=1)
    add(17, 1, [-5], [6], hold=1)

    return cases


def random_cases(rng: random.Random, start_id: int, count: int) -> list[dict]:
    cases: list[dict] = []
    for offset in range(count):
        case_id = start_id + offset
        k = rng.randint(1, 64)
        a = [rng.randint(-128, 127) for _ in range(k)]
        b = [rng.randint(-128, 127) for _ in range(k)]
        gap = rng.randint(0, 3)
        hold = rng.randint(1, 3)
        cases.append(make_case(case_id, k, a, b, gap, hold))
    return cases


def write_vector_file(path: Path, seed: int, cases: list[dict]) -> None:
    lines: list[str] = [
        f"# seed={seed} num_cases={len(cases)}",
        "# format: CASE id K gap hold expected",
        "#         BEAT a0 a1 a2 a3 b0 b1 b2 b3 keep",
        "#         END",
        "",
    ]
    for case in cases:
        lines.append(
            "CASE {case_id} {k} {gap} {hold} {expected}".format(**case)
        )
        for av, bv, keep in case["beats"]:
            keep_bits = f"{keep:04b}"
            lines.append(
                "BEAT {a0} {a1} {a2} {a3} {b0} {b1} {b2} {b3} {keep}".format(
                    a0=av[0],
                    a1=av[1],
                    a2=av[2],
                    a3=av[3],
                    b0=bv[0],
                    b1=bv[1],
                    b2=bv[2],
                    b3=bv[3],
                    keep=keep_bits,
                )
            )
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--random-count", type=int, default=103, help="number of random cases")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dot_product_parallel_vectors.txt",
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

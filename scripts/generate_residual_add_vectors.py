#!/usr/bin/env python3
"""D11: Generate residual_add_vectors.txt."""

from __future__ import annotations

import random
from pathlib import Path

SEED = 20260905


def sat_add(x: int, y: int) -> int:
    s = int(x) + int(y)
    if s > 127:
        return 127
    if s < -127:
        return -127
    return s


def build_cases(rng: random.Random) -> list[dict]:
    cases: list[dict] = []
    specs = [
        (1, [(100, 20), (-100, -20), (127, 1), (-127, -1), (64, 64), (-64, -64)]),
        (2, [(127, 127), (-127, -127), (127, -127), (-127, 127), (50, 80), (-50, -80)]),
        (3, [(0, 0), (1, -1), (127, 0), (-127, 0), (0, 127), (0, -127)]),
        (4, [(rng.randint(-127, 127), rng.randint(-127, 127)) for _ in range(8)]),
        (5, [(rng.randint(-127, 127), rng.randint(-127, 127)) for _ in range(16)]),
    ]
    for cid, pairs in specs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        zs = [sat_add(x, y) for x, y in pairs]
        cases.append({"case_id": cid, "len": len(pairs), "x": xs, "y": ys, "z": zs})
    return cases


def main() -> None:
    rng = random.Random(SEED)
    cases = build_cases(rng)
    out = Path(__file__).resolve().parent.parent / "residual_add_vectors.txt"
    lines = [f"# seed={SEED} num_cases={len(cases)}", ""]
    for c in cases:
        lines.append(f"CASE {c['case_id']} {c['len']}")
        lines.append("X")
        lines.extend(str(v) for v in c["x"])
        lines.append("Y")
        lines.extend(str(v) for v in c["y"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in c["z"])
        lines.append("END")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()

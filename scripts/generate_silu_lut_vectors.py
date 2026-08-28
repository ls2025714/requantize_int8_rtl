#!/usr/bin/env python3
"""D11: Generate silu_lut_vectors.txt."""

from __future__ import annotations

import random
from pathlib import Path

from silu_fixed_ref import SILU_LUT, SILU_LUT_MAX, SILU_LUT_MIN, silu_int8

SEED = 20260906


def main() -> None:
    rng = random.Random(SEED)
    cases: list[dict] = []
    # case 1: all LUT entries spot check (sample every 8)
    cases.append({"case_id": 1, "inputs": list(range(SILU_LUT_MIN, SILU_LUT_MAX + 1, 8))})
    # case 2: boundary
    cases.append({"case_id": 2, "inputs": [-128, -127, -1, 0, 1, 126, 127]})
    # case 3: random
    cases.append({"case_id": 3, "inputs": [rng.randint(-128, 127) for _ in range(32)]})
    for c in cases:
        c["outputs"] = [silu_int8(v) for v in c["inputs"]]
    out = Path(__file__).resolve().parent.parent / "silu_lut_vectors.txt"
    lines = [f"# seed={SEED} num_cases={len(cases)}", ""]
    for c in cases:
        n = len(c["inputs"])
        lines.append(f"CASE {c['case_id']} {n}")
        lines.append("IN")
        lines.extend(str(v) for v in c["inputs"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in c["outputs"])
        lines.append("END")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases ({sum(len(c['inputs']) for c in cases)} elems) to {out}")


if __name__ == "__main__":
    main()

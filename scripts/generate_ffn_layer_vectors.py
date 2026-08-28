#!/usr/bin/env python3
"""D11: Generate ffn_layer_vectors.txt for layered gate/up/down TB."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_mini_gpt_ffn import D_FF, EMBD, export_all, weight_tile_major_flat_nk
from export_mini_gpt_wq_slice import DEFAULT_CHECKPOINT, SHIFT_BITS

SEED = 20260907


def write_vectors(path: Path, seed: int, cases: list[dict], source: str) -> None:
    lines = [
        f"# seed={seed} num_cases={len(cases)}",
        f"# source={source}",
        "# CASE id stage M",
        "# A / WEIGHT / MULT / EXPECT / END",
        "",
    ]
    for c in cases:
        lines.append(f"CASE {c['case_id']} {c['stage']} {c['m']}")
        lines.append("A")
        lines.extend(str(v) for v in c["a"])
        lines.append("WEIGHT")
        lines.extend(str(v) for v in c["weights"])
        lines.append("MULT")
        lines.extend(str(v) for v in c["mults"])
        lines.append("EXPECT")
        lines.extend(str(v) for v in c["expected"])
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument(
        "--learn-root",
        type=Path,
        default=Path(r"F:/Users/22563/Desktop/learn"),
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "ffn_layer_vectors.txt",
    )
    args = ap.parse_args()
    cases, hdr = export_all(args.checkpoint, args.learn_root, 32, args.seed)
    write_vectors(args.output, args.seed, cases, hdr["source"])
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()

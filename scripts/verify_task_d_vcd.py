#!/usr/bin/env python3
"""Task D：解析 task_d_full.vcd，与 dot_product_parallel_vectors.txt 对拍。"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VCD = REPO / "task_d_full.vcd"
VECTORS = REPO / "dot_product_parallel_vectors.txt"

KEYS = {
    "<": "m_valid",
    ">": "m_result",
    "?": "test_count",
    "@": "mismatch_count",
    "3": "state",
    "4": "valid_s1",
    "5": "valid_s2",
    "6": "valid_s3",
    "7": "valid_s4",
    ":": "acc_valid",
    ";": "acc_value",
    "8": "partial_valid",
    "9": "partial_sum",
    "$": "cmd_ready",
    "&": "s_valid",
    "'": "s_ready",
}


def load_golden(path: Path) -> dict[int, int]:
    golden: dict[int, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith("CASE "):
            parts = line.split()
            golden[int(parts[1])] = int(parts[5])
    return golden


def apply_vcd_line(line: str, vals: dict[str, int | None]) -> None:
    if line[0] in "01":
        sid = line[1:]
        if sid in KEYS:
            vals[sid] = int(line[0])
    elif line.startswith("b"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in KEYS:
            vals[parts[1]] = (
                int(parts[0][1:], 2) if parts[0] != "bx" else None
            )


def parse_vcd(path: Path) -> tuple[int, list, list, int]:
    vals = {k: None for k in KEYS}
    prev_m_valid = 0
    m_valid_rises: list[tuple] = []
    mismatch_nonzero: list[tuple] = []
    drain_acc_samples = 0
    t = 0
    pending: list[str] = []

    def flush_timestamp() -> None:
        nonlocal prev_m_valid, drain_acc_samples
        for ch in pending:
            apply_vcd_line(ch, vals)
        pending.clear()

        if vals["<"] == 1 and prev_m_valid == 0:
            m_valid_rises.append(
                (t, vals[">"], vals["?"], vals["@"], vals["3"])
            )
        if vals["@"] is not None and vals["@"] != 0:
            mismatch_nonzero.append((t, vals["@"]))
        if vals.get("3") == 2 and vals.get(":") == 1:
            drain_acc_samples += 1

        prev_m_valid = vals["<"] if vals["<"] is not None else prev_m_valid

    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if pending:
                flush_timestamp()
            t = int(line[1:])
            continue
        if not line or line.startswith("$"):
            continue
        pending.append(line)

    if pending:
        flush_timestamp()

    return t, m_valid_rises, mismatch_nonzero, drain_acc_samples


def signed32(v: int | None) -> int | None:
    if v is None:
        return None
    return v if v < 0x80000000 else v - 0x100000000


def main() -> int:
    if not VCD.is_file():
        print(f"ERROR: missing {VCD}")
        return 1

    golden = load_golden(VECTORS)
    duration, rises, mmc_events, drain_acc = parse_vcd(VCD)

    print(f"VCD: {VCD.name} ({VCD.stat().st_size} bytes)")
    print(f"Simulation duration: {duration / 1000:.0f} ns")
    print(f"m_valid rising edges: {len(rises)}")
    print(f"mismatch_count != 0 events: {len(mmc_events)}")
    print(f"DRAIN + acc_valid samples: {drain_acc}")

    mismatches = []
    for i, (time_ns, mres, tc, mmc, state) in enumerate(rises, start=1):
        signed = signed32(mres)
        exp = golden.get(i)
        ok = exp is not None and signed == exp and mmc == 0 and state == 3
        if not ok:
            mismatches.append((i, time_ns, signed, exp, tc, mmc, state))

    states_at_output = {x[4] for x in rises}
    print(f"state at m_valid (expect OUTPUT=3 only): {states_at_output}")

    if len(rises) != 120:
        print(f"WARN: expected 120 m_valid edges, got {len(rises)}")

    print("\nSample checks:")
    for idx in [1, 2, 3, 14, 58, 59, 120]:
        if idx <= len(rises):
            t, mres, tc, mmc, state = rises[idx - 1]
            signed = signed32(mres)
            exp = golden[idx]
            print(
                f"  case {idx}: t={t / 1000:.0f}ns result={signed} exp={exp} "
                f"tc={tc} mmc={mmc} state={state}"
            )

    print(f"\nVCD vs golden mismatches: {len(mismatches)}")
    for row in mismatches[:20]:
        print(" ", row)

    if mismatches or mmc_events or states_at_output != {3}:
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS — 120/120 m_valid results match Python golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())

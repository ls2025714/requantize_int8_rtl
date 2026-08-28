#!/usr/bin/env python3
"""D3：解析 gemm_parallel_full.vcd，与 gemm_parallel_vectors.txt INT32 期望对拍。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VCD = REPO / "gemm_parallel_full.vcd"
VECTORS = REPO / "gemm_parallel_vectors.txt"

# VCD id -> name (from gemm_parallel_full.vcd header)
KEYS = {
    "!": "clk",
    '"': "rst_n",
    "#": "cmd_valid",
    "$": "cmd_ready",
    "%": "cmd_m",
    "&": "cmd_n",
    "'": "cmd_k",
    ".": "c_valid",
    "/": "c_ready",
    "0": "c_data",
    "1": "c_row",
    "2": "c_col",
    ">": "test_count",
    "?": "mismatch_count",
    "3": "state",
    "4": "row_index",
    "5": "col_index",
    "6": "beat_index",
    "7": "dot_s_valid",
    "8": "dot_s_ready",
    "9": "dot_s_keep",
    ":": "dot_m_valid",
    ";": "dot_m_ready",
    "<": "dot_m_result",
    "=": "dot_state",
}


def parse_vectors(path: Path) -> list[dict]:
    cases: list[dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("CASE "):
            i += 1
            continue
        parts = line.split()
        case_id = int(parts[1])
        m, n, k = int(parts[2]), int(parts[3]), int(parts[4])
        i += 1
        assert lines[i].strip() == "A"
        i += 1
        a = [int(lines[i + j]) for j in range(m * k)]
        i += m * k
        assert lines[i].strip() == "B"
        i += 1
        b = [int(lines[i + j]) for j in range(k * n)]
        i += k * n
        assert lines[i].strip() == "EXPECT"
        i += 1
        expect = [int(lines[i + j]) for j in range(m * n)]
        i += m * n
        assert lines[i].strip() == "END"
        i += 1
        outputs = []
        for row in range(m):
            for col in range(n):
                outputs.append({"row": row, "col": col, "expected": expect[row * n + col]})
        cases.append({"case_id": case_id, "m": m, "n": n, "k": k, "outputs": outputs})
    return cases


def to_signed32(value: int | None) -> int | None:
    if value is None:
        return None
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def apply_line(line: str, vals: dict[str, int | None]) -> None:
    if not line:
        return
    if line[0] in "01":
        sid = line[1:]
        if sid in KEYS:
            vals[sid] = int(line[0])
    elif line.startswith("b"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in KEYS:
            if parts[0] in ("bx", "bz"):
                vals[parts[1]] = None
            else:
                vals[parts[1]] = int(parts[0][1:], 2)


def parse_vcd_outputs(path: Path) -> tuple[list[dict], list[tuple[int, int]], int]:
    vals: dict[str, int | None] = {k: None for k in KEYS}
    prev_c_valid = 0
    prev_c_ready = 0
    outputs: list[dict] = []
    mismatch_events: list[tuple[int, int]] = []
    t = 0
    pending: list[str] = []

    def flush() -> None:
        nonlocal prev_c_valid, prev_c_ready
        for ch in pending:
            apply_line(ch, vals)
        pending.clear()

        c_valid = vals.get(".") or 0
        c_ready = vals.get("/") or 0
        mismatch = vals.get("?") or 0

        if mismatch:
            mismatch_events.append((t, mismatch))

        # Sample when c_valid rises (new output available)
        if c_valid == 1 and prev_c_valid == 0:
            outputs.append(
                {
                    "time_ps": t,
                    "c_data": to_signed32(vals.get("0")),
                    "c_row": vals.get("1"),
                    "c_col": vals.get("2"),
                    "state": vals.get("3"),
                    "dot_s_keep": vals.get("9"),
                    "beat_index": vals.get("6"),
                }
            )

        prev_c_valid = c_valid
        prev_c_ready = c_ready

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            flush()
            t = int(line[1:])
        elif line.startswith("b") or line[:1] in "01":
            pending.append(line)

    flush()
    final_test_count = vals.get(">") or 0
    return outputs, mismatch_events, final_test_count


def main() -> int:
    if not VCD.is_file():
        print(f"ERROR: missing {VCD}")
        return 1
    if not VECTORS.is_file():
        print(f"ERROR: missing {VECTORS}")
        return 1

    cases = parse_vectors(VECTORS)
    expected_total = sum(len(c["outputs"]) for c in cases)
    vcd_outputs, mismatch_events, final_test_count = parse_vcd_outputs(VCD)

    print(f"VCD: {VCD} ({VCD.stat().st_size} bytes)")
    print(f"vector cases: {len(cases)}, expected C outputs: {expected_total}")
    print(f"VCD c_valid rise events: {len(vcd_outputs)}")
    print(f"final test_count (VCD): {final_test_count}")

    errors = 0
    if len(vcd_outputs) != expected_total:
        print(f"FAIL: output count {len(vcd_outputs)} != expected {expected_total}")
        errors += 1

    idx = 0
    for case in cases:
        for out in case["outputs"]:
            if idx >= len(vcd_outputs):
                break
            got = vcd_outputs[idx]
            exp = out["expected"]
            rtl = got["c_data"]
            if rtl != exp or got["c_row"] != out["row"] or got["c_col"] != out["col"]:
                errors += 1
                print(
                    f"FAIL case={case['case_id']} C[{out['row']}][{out['col']}] "
                    f"rtl={rtl} expected={exp} coord=({got['c_row']},{got['c_col']}) t={got['time_ps']}ps"
                )
            idx += 1

    if errors == 0 and idx == expected_total:
        print(f"PASS: all {expected_total} C outputs match golden")
        print(f"PASS: mismatch_count never nonzero ({len(mismatch_events)} events)")

    # Spot-check case 1 FSM / keep for K=3 (first case, first dot product)
    case1_first = next((o for o in vcd_outputs if o["c_data"] == 58), None)
    if case1_first:
        print(
            f"Case1 C[0][0]=58 @ {case1_first['time_ps']}ps, "
            f"state={case1_first['state']} (OUTPUT_C=6 expected at output)"
        )

    if mismatch_events:
        print(f"FAIL: mismatch_count nonzero at: {mismatch_events[:5]}")
        errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

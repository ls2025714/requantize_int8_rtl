#!/usr/bin/env python3
"""Sanity-check tiled_head0_q_full.vcd for D6 16-tile behavior."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

VCD = Path(__file__).resolve().parent.parent / "tiled_head0_q_full.vcd"

REQUIRED_SUBSTR = [
    "clk",
    "rst_n",
    "cmd_valid",
    "c_valid",
    "c_data",
    "c_row",
    "c_col",
    "tile_state",
    "n_tile",
    "k_tile",
    "tile_idx",
    "k_base",
    "n_base",
    "replay_start",
    "replay_base_addr",
    "test_count",
    "mismatch_count",
]


def to_int(raw: str) -> int | None:
    raw = raw.lower()
    if any(c in raw for c in "xz"):
        return None
    if set(raw) <= set("01"):
        return int(raw, 2)
    return int(raw)


def parse_vcd(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    id_to_name: dict[str, str] = {}
    for m in re.finditer(
        r"\$var\s+\w+\s+\d+\s+(\S+)\s+(\S+)(?:\s+\[[^\]]+\])?\s+\$end", text
    ):
        id_to_name[m.group(1)] = m.group(2)

    changes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    times: list[int] = []
    cur_t = 0
    parts = text.split("$enddefinitions $end", 1)
    if len(parts) < 2:
        raise RuntimeError("missing $enddefinitions")
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("#"):
            cur_t = int(line[1:])
            times.append(cur_t)
            continue
        if line[0] in "01xXzZ":
            changes[line[1:]].append((cur_t, line[0]))
        elif line[0] in "bBrR":
            bits, vid = line.split()
            changes[vid].append((cur_t, bits[1:]))
    return id_to_name, changes, times


def find_id(id_to_name: dict[str, str], key: str) -> str | None:
    # exact
    for vid, name in id_to_name.items():
        if name == key:
            return vid
    # suffix match (dut.tile_idx / tile_idx)
    for vid, name in id_to_name.items():
        if name.endswith(key) or name.split(".")[-1] == key:
            return vid
    return None


def unique_ints(series: list[tuple[int, str]]) -> list[int]:
    vals: set[int] = set()
    for _, raw in series:
        v = to_int(raw)
        if v is not None:
            vals.add(v)
    return sorted(vals)


def final_int(series: list[tuple[int, str]]) -> int | None:
    last = None
    for _, raw in series:
        v = to_int(raw)
        if v is not None:
            last = v
    return last


def rising_edges(series: list[tuple[int, str]]) -> int:
    rises = 0
    prev = "0"
    for _, raw in series:
        if prev == "0" and raw == "1":
            rises += 1
        if raw in "01":
            prev = raw
    return rises


def main() -> None:
    if not VCD.exists():
        raise SystemExit(f"missing VCD: {VCD}")

    size = VCD.stat().st_size
    id_to_name, changes, times = parse_vcd(VCD)
    print(f"VCD: {VCD}")
    print(f"size: {size} bytes")
    print(f"signals: {len(id_to_name)}")
    print(f"time_points: {len(times)}  end=#{times[-1] if times else 'N/A'}")
    print("signal names (first 25):")
    for name in list(id_to_name.values())[:25]:
        print(f"  {name}")

    missing = [k for k in REQUIRED_SUBSTR if find_id(id_to_name, k) is None]
    print(f"required found: {len(REQUIRED_SUBSTR) - len(missing)}/{len(REQUIRED_SUBSTR)}")
    if missing:
        print("MISSING:", ", ".join(missing))

    def series(key: str) -> list[tuple[int, str]]:
        vid = find_id(id_to_name, key)
        return changes.get(vid, []) if vid else []

    tile_vals = unique_ints(series("tile_idx"))
    k_vals = unique_ints(series("k_base"))
    n_vals = unique_ints(series("n_base"))
    rb_vals = unique_ints(series("replay_base_addr"))
    test_final = final_int(series("test_count"))
    mm_final = final_int(series("mismatch_count"))
    c_rises = rising_edges(series("c_valid"))

    print(f"tile_idx unique: {tile_vals}")
    print(f"k_base unique:   {k_vals}")
    print(f"n_base unique:   {n_vals}")
    print(f"replay_base unique count={len(rb_vals)} values={rb_vals}")
    print(f"test_count final={test_final}  mismatch_count final={mm_final}")
    print(f"c_valid rising edges={c_rises} (expect 176 = 11*16)")

    expect_tiles = list(range(16))
    expect_k = [0, 16, 32, 48]
    expect_n = [0, 4, 8, 12]
    expect_rb = list(range(0, 1024, 64))

    checks = {
        "size>4KB": size > 4096,
        "no_missing_signals": not missing,
        "tile_idx_0..15": tile_vals == expect_tiles or set(expect_tiles).issubset(tile_vals),
        "k_base_tiles": set(expect_k).issubset(k_vals),
        "n_base_tiles": set(expect_n).issubset(n_vals),
        "replay_bases": set(expect_rb).issubset(rb_vals),
        "test_count==4": test_final == 4,
        "mismatch==0": mm_final == 0,
        "c_valid_rises==176": c_rises == 176,
    }
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")

    ok = all(checks.values())
    print("VERDICT:", "PASS — VCD content matches D6 16-tile run" if ok else "FAIL — see checks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How many columns of the planet field pass through more than one planet?

`isekai_api:world_floor` walks a column from the heightmap downwards. Before
isekai-api 2.2.0 (`35f020b`) it reported only the topmost body in the column, so
every planet with another planet above it in the same column lost its
floor-anchored ore. This script measures how many columns of `planet_field.json`
are in that situation, which is what decides whether 2.2.0's per-body resolution
changes anything for Planet World.

Two numbers, because worldgen does not sample the density function per block:

  exact      column (x, z) is inside a sphere's XZ disk at its nominal radius
  +margin    the same with `--margin` blocks added to every radius. Terrain is
             interpolated between noise cells (`size_horizontal=1` => 4 blocks in
             XZ), so a sphere's real footprint can bulge outward by up to one
             cell width. The margin is the worst case, not the expected case.

Enumerates one full period cell of integer columns, counting every periodic image
of every sphere. Radii and centres come from `verify_cosmos_layout.load_spheres()`
so this cannot drift from the layout it measures.

Usage: python tools/check_column_crossing.py [--margin 4] [--list 10]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_cosmos_layout import load_spheres  # noqa: E402

SHIFTS = range(-2, 3)


def images(spheres, margin: float):
    """Every periodic image as (name, cx, cz, radius, cy)."""
    out = []
    for s in spheres:
        shifts = SHIFTS if s.period is not None else (0,)
        for sx, sz in itertools.product(shifts, shifts):
            out.append(
                (
                    s.name,
                    s.cx + sx * (s.period or 0.0),
                    s.cz + sz * (s.period or 0.0),
                    s.r_out + margin,
                    s.cy,
                )
            )
    return out


def column_hits(imgs, x: float, z: float):
    """Which images own column (x, z), as (name, cy, radius, half_height)."""
    hits = []
    for name, cx, cz, r, cy in imgs:
        d2 = (x - cx) ** 2 + (z - cz) ** 2
        if d2 < r * r:
            hits.append((name, cy, r, math.sqrt(r * r - d2)))
    return hits


def scan(spheres, margin: float, period: float, step: int = 1):
    """Fraction of columns in one period cell that pass through >= 2 bodies."""
    imgs = images(spheres, margin)
    total = 0
    multi = 0
    worst: list[tuple[int, int, list]] = []
    n = int(period)
    for x in range(0, n, step):
        for z in range(0, n, step):
            total += 1
            hits = column_hits(imgs, x, z)
            if len(hits) >= 2:
                multi += 1
                worst.append((x, z, hits))
    return total, multi, worst


def pair_gaps(spheres, margin: float):
    """Min XZ surface gap for every pair and every self-image pair."""
    rows = []
    for a, b in itertools.combinations(spheres, 2):
        rows.append((min_gap(a, b, margin), a.name, b.name))
    for a in spheres:
        if a.period is not None:
            rows.append((min_gap(a, a, margin), a.name, "(own periodic image)"))
    rows.sort()
    return rows


def min_gap(a, b, margin: float) -> float:
    shifts_a = SHIFTS if a.period is not None else (0,)
    shifts_b = SHIFTS if b.period is not None else (0,)
    best = math.inf
    for sxa, sza in itertools.product(shifts_a, shifts_a):
        ax = a.cx + sxa * (a.period or 0.0)
        az = a.cz + sza * (a.period or 0.0)
        for sxb, szb in itertools.product(shifts_b, shifts_b):
            if a is b and (sxa, sza) == (sxb, szb):
                continue
            bx = b.cx + sxb * (b.period or 0.0)
            bz = b.cz + szb * (b.period or 0.0)
            gap = math.hypot(ax - bx, az - bz) - (a.r_out + margin) - (b.r_out + margin)
            best = min(best, gap)
    return best


def verify_in_world(worst, host: str, port: int, password: str) -> int:
    """Ask a running server whether the +margin candidates really cross two bodies.

    Non-destructive: only `execute if block`. A candidate counts as a real
    crossing when at least one probe Y inside each of two different bodies is
    non-air. Runs a positive control first (the home planet's centre must be
    solid), because a probe that answers "air" to everything would otherwise
    report a clean 0.
    """
    from probe_planet_ores import Rcon  # noqa: PLC0415 - only needed in rcon mode

    rcon = Rcon(host, port, password)
    print(f"\nseed reported by the server: {rcon.cmd('seed').strip()}")

    control = rcon.cmd(
        "execute if block 0 100 0 #minecraft:base_stone_overworld run time query gametime"
    )
    solid_control = "The time is" in control
    print(f"positive control: home centre (0,100,0) is stone -> {solid_control}")
    if not solid_control:
        print("  probe cannot see solid blocks; the 0 below would be meaningless")
        rcon.close()
        return 1

    xs = [x for x, _, _ in worst]
    zs = [z for _, z, _ in worst]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    rcon.cmd(f"forceload add {x0} {z0} {x1} {z1}")

    crossings = 0
    for x, z, hits in worst:
        solid_bodies = 0
        for _, cy, _, half in hits:
            probes = {int(cy), int(cy - half / 2), int(cy + half / 2)}
            if any(
                "The time is"
                in rcon.cmd(
                    f"execute unless block {x} {y} {z} minecraft:air "
                    f"run time query gametime"
                )
                for y in probes
            ):
                solid_bodies += 1
        if solid_bodies >= 2:
            crossings += 1
            if crossings <= 5:
                print(f"  CROSSING at column ({x},{z})")
    rcon.cmd(f"forceload remove {x0} {z0} {x1} {z1}")
    rcon.close()
    print(
        f"in-world: {crossings} / {len(worst)} of the +margin candidates really "
        "pass through two bodies"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--margin",
        type=float,
        default=4.0,
        help="blocks added to every radius (default 4 = one noise cell in XZ)",
    )
    ap.add_argument("--list", type=int, default=8, help="how many columns to print")
    ap.add_argument(
        "--rcon",
        action="store_true",
        help="check the +margin candidates against a running dedicated server",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25599)
    ap.add_argument("--password", default="planetworld")
    args = ap.parse_args()

    spheres = load_spheres()
    periods = {s.period for s in spheres if s.period is not None}
    if len(periods) != 1:
        print(f"expected a single period, got {periods}")
        return 1
    period = periods.pop()

    for margin in (0.0, args.margin):
        tag = "exact" if margin == 0 else f"+{margin:g} margin"
        total, multi, worst = scan(spheres, margin, period)
        pct = 100.0 * multi / total
        print(f"\n{tag}: {multi} / {total} columns hit >= 2 bodies  ({pct:.4f}%)")
        gaps = pair_gaps(spheres, margin)
        neg = [g for g in gaps if g[0] < 0]
        print(f"  pairs with negative XZ surface gap: {len(neg)} / {len(gaps)}")
        for g, n1, n2 in gaps[:3]:
            print(f"    {g:8.2f}  {n1}  vs  {n2}")
        for x, z, hits in worst[: args.list]:
            names = ", ".join(
                f"{n.split('.')[-2] if '.' in n else n}@y{cy:.0f}" f"(+-{h:.1f})"
                for n, cy, _, h in sorted(hits, key=lambda t: -t[1])
            )
            print(f"    column ({x},{z}): {names}")
        if margin and args.rcon:
            return verify_in_world(worst, args.host, args.port, args.password)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

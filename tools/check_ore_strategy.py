#!/usr/bin/env python3
"""Desk-check the ore_strategy column mapping against the 16 planet_world ore
placed_features, for both the old (blocks) and current (proportional) scale.

Reproduces the RemapStrategy.ColumnLocal / ColumnBand arithmetic in
mod-029-isekai-api (api/remap/RemapStrategy.java, api/remap/ColumnBand.java)
without touching the API module. Reads the ore Y ranges straight from this
mod's own placed_feature JSONs (never hand-copied) and the planet radii from
worldgen/density_function/planet_field.json (read-only in this script — never
overwritten), so a future ore or planet-radius change re-runs the same check.

Usage: python tools/check_ore_strategy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACED = ROOT / "src/main/resources/data/planet_world/worldgen/placed_feature"
WORLDSHAPE = ROOT / "src/main/resources/data/planet_world/isekai/worldshape/cosmos.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_cosmos_layout import load_spheres  # noqa: E402 - reuse the vetted parser,

# never hand-copy planet radii (verify_cosmos_layout.load_spheres() already reads
# planet_field.json + the relief bounds and derives the relief-adjusted r_out/r_in
# per sphere; that is strictly better evidence than a literal here).


def _reference_column() -> tuple[int, int]:
    """The ore_strategy's vanilla reference levels, read from cosmos.json.

    Absent fields take RemapStrategy.ColumnLocal's own defaults (64 / -64), so this
    tracks the JSON instead of restating it — if the worldshape ever names its own
    surface_y/floor_y, the collapse check below re-runs against the real values.
    """
    strategy = json.loads(WORLDSHAPE.read_text(encoding="utf-8"))["ore_strategy"]
    return strategy.get("surface_y", 64), strategy.get("floor_y", -64)


SURFACE_Y, FLOOR_Y = _reference_column()
REFERENCE_THICKNESS = 128  # ColumnBand.VANILLA_THICKNESS (unused by "proportional")
DEPTH_EPSILON = 1.0e-9  # ColumnBand.DEPTH_EPSILON

ORES = [
    "ore_coal",
    "ore_coal_rich",
    "ore_copper",
    "ore_copper_rich",
    "ore_iron",
    "ore_iron_rich",
    "ore_iron_deep",
    "ore_diamond",
    "ore_diamond_rich",
    "ore_emerald",
    "ore_gold",
    "ore_gold_rich",
    "ore_lapis",
    "ore_lapis_rich",
    "ore_redstone",
    "ore_redstone_rich",
]


def read_y_range(name: str) -> tuple[int, int]:
    data = json.loads((PLACED / f"{name}.json").read_text(encoding="utf-8"))
    for p in data["placement"]:
        if p["type"] == "minecraft:height_range":
            h = p["height"]
            return h["min_inclusive"]["absolute"], h["max_inclusive"]["absolute"]
    raise ValueError(f"{name}: no height_range placement")


def depth_of(y: int) -> float:
    d = (SURFACE_Y - y) / (SURFACE_Y - FLOOR_Y)
    return max(0.0, min(1.0, d))


def anchored_to_top(from_depth: float, to_depth: float) -> bool:
    return (from_depth + to_depth) / 2.0 <= 0.5


def resolve_blocks(top_y: int, bottom_y: int, depth: float) -> int:
    if anchored_to_top(depth, depth):  # per-sample; caller passes one depth at a time
        return top_y - round(depth * REFERENCE_THICKNESS)
    return bottom_y + round((1.0 - depth) * REFERENCE_THICKNESS)


def resolve_proportional(top_y: int, bottom_y: int, depth: float) -> int:
    return top_y - round(depth * (top_y - bottom_y))


def collapsed_at_anchor(from_depth: float, to_depth: float) -> bool:
    """Mirror of ColumnBand.collapsedAtAnchor().

    A band of zero width pinned to depth 0.0 or 1.0 is one whose whole vanilla Y range
    fell outside SURFACE_Y..FLOOR_Y and was flattened by depth_of()'s clamp. Every sample
    then resolves to the same plane at the very edge of the body, which for an ore reads
    as "does not spawn" -- the Sky World failure (isekai-api issue #2). A zero-width band
    away from both anchors is a legitimate single-plane source range and is not reported.
    """
    if to_depth - from_depth > DEPTH_EPSILON:
        return False
    return from_depth <= DEPTH_EPSILON or from_depth >= 1.0 - DEPTH_EPSILON


def check_no_collapsed_bands(bands: list) -> int:
    print()
    print(
        f"--- collapsed-band check (reference column "
        f"surface_y={SURFACE_Y}, floor_y={FLOOR_Y}) ---"
    )
    bad = [
        (name, min_y, max_y, fd)
        for name, min_y, max_y, fd, td in bands
        if collapsed_at_anchor(fd, td)
    ]
    if not bad:
        print(
            f"OK: all {len(bands)} ores keep a band with width; none collapsed onto an anchor"
        )
        return 0
    for name, min_y, max_y, fd in bad:
        print(
            f"FAIL: {name} (vanilla Y {min_y}..{max_y}) collapses to zero width at depth "
            f"{fd:.1f} -- entirely outside surface_y={SURFACE_Y}..floor_y={FLOOR_Y}, so it "
            "resolves to one plane at the edge of the body and effectively never spawns. "
            "Widen the ore_strategy's surface_y/floor_y to cover this range."
        )
    return 1


def main() -> int:
    bands = []
    for name in ORES:
        min_y, max_y = read_y_range(name)
        from_depth, to_depth = depth_of(max_y), depth_of(min_y)  # shallow..deep
        bands.append((name, min_y, max_y, from_depth, to_depth))

    print(f"{'ore':<18}{'vanilla Y':>12}  {'from_depth':>10}  {'to_depth':>10}  group")
    for name, min_y, max_y, fd, td in bands:
        group = "shallow(top)" if anchored_to_top(fd, td) else "deep(bottom)"
        print(f"{name:<18}{min_y:>5}..{max_y:<5}  {fd:>10.4f}  {td:>10.4f}  {group}")

    failures = check_no_collapsed_bands(bands)

    shallow_deepest = max(
        td for _, _, _, fd, td in bands if anchored_to_top(fd, td) for td in [td]
    )
    deep_shallowest = min(
        fd for _, _, _, fd, td in bands if not anchored_to_top(fd, td) for fd in [fd]
    )
    print(f"\nshallow group deepest to_depth = {shallow_deepest:.4f}")
    print(f"deep group shallowest from_depth = {deep_shallowest:.4f}")
    print(
        f"margin between groups = {deep_shallowest - shallow_deepest:.4f} (>0 means no overlap is possible under proportional scale, for any thickness)"
    )
    print(
        f"note: under proportional scale the two groups are separated by "
        f"{deep_shallowest - shallow_deepest:.4f} * thickness blocks, so at thickness "
        f"< ~{1 / (deep_shallowest - shallow_deepest):.0f} the two groups round onto the "
        "same Y (an inherent limit of a physically thin rim, not a strategy defect — "
        "ColumnRelativeModifier already skips columns with topY-bottomY < 2 outright)."
    )

    spheres = load_spheres()
    print(
        "\n--- per-planet thickness, from verify_cosmos_layout.load_spheres() "
        "(r_out includes the relief bound, which is <= 0 here so r_out == r) ---"
    )
    for s in sorted(spheres, key=lambda s: s.r_out, reverse=True):
        thickness = 2 * s.r_out
        print(
            f"{s.name:<12} R={s.r_out:>5.1f}  max thickness={thickness:>5.1f}", end="  "
        )
        overlap_blocks = thickness <= (shallow_deepest * REFERENCE_THICKNESS) + (
            REFERENCE_THICKNESS - deep_shallowest * REFERENCE_THICKNESS
        )
        void_blocks = thickness < shallow_deepest * REFERENCE_THICKNESS
        print(f"blocks-mode: overlap={overlap_blocks}  iron_deep-in-void={void_blocks}")

    max_thickness = max(2 * s.r_out for s in spheres)
    needed_scan = max_thickness + 1  # WorldFloor: 1 leading-air step + the body itself
    print(
        f"\nmax possible column thickness across every planet/image = {max_thickness:.1f}"
        f"  ->  WorldFloor needs max_scan >= {needed_scan:.0f}"
        f"  (default DEFAULT_MAX_SCAN=128, headroom={128 - needed_scan:.0f})"
    )
    print(
        "verify_cosmos_layout.py's own XZ-occlusion check (min_xz_surface_gap over all "
        "periodic images, all pairs) is the authority that no two spheres ever share a "
        "column, so no column can exceed a single sphere's own thickness — see its "
        "'OK: no ... XZ-occlusion ... violations' output."
    )

    print(
        "\n--- group span at R=42 dead centre (thickness=84, topY=42, bottomY=-42) ---"
    )
    ty, by = 42, -42
    for name, min_y, max_y, fd, td in bands:
        if anchored_to_top(fd, td):
            top_of_band = resolve_blocks(ty, by, fd)
            bottom_of_band = resolve_blocks(ty, by, td)
        else:
            top_of_band = by + round((1.0 - fd) * REFERENCE_THICKNESS)
            bottom_of_band = by + round((1.0 - td) * REFERENCE_THICKNESS)
        top_prop = resolve_proportional(ty, by, fd)
        bottom_prop = resolve_proportional(ty, by, td)
        print(
            f"{name:<18} blocks: y={bottom_of_band:>4}..{top_of_band:<4}"
            f"   proportional: y={bottom_prop:>4}..{top_prop:<4}"
        )

    print(
        "\n--- worked example: thinnest planet's dead centre (R=16, thickness=32) ---"
    )
    top_y, bottom_y = 32, 0  # arbitrary column with thickness 32
    for name, min_y, max_y, fd, td in bands:
        mid_depth = (fd + td) / 2.0
        y_blocks = (
            resolve_blocks(top_y, bottom_y, mid_depth)
            if anchored_to_top(fd, td)
            else bottom_y + round((1.0 - mid_depth) * REFERENCE_THICKNESS)
        )
        y_prop = resolve_proportional(top_y, bottom_y, mid_depth)
        in_body_blocks = bottom_y <= y_blocks <= top_y
        in_body_prop = bottom_y <= y_prop <= top_y
        print(
            f"{name:<18} blocks->y={y_blocks:>5} in_body={in_body_blocks}   proportional->y={y_prop:>5} in_body={in_body_prop}"
        )

    return failures


if __name__ == "__main__":
    sys.exit(main())

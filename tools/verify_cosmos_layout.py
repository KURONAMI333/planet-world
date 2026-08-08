#!/usr/bin/env python3
"""Cosmos planet-field layout regression check.

Parses the live density function tree at
`data/planet_world/worldgen/density_function/planet_field.json` (a nested
`isekai_api:max` of spheres, each `isekai_api:add(constant r, negate(distance(...)))`,
optionally wrapped in `isekai_api:repeat(period_x, period_z, f=...)`, the whole tree
summed with the seeded relief term `planet_world:planet_relief`) and checks,
against the *actual JSON values* (not a hand-copied table, so this fails the moment
the JSON drifts from what was verified):

  1. Tile-boundary safety: every tiled sphere's center, reduced mod its repeat
     period, stays at least (radius + BOUNDARY_MARGIN) away from every tile edge.
     This is the fix for the "vertical flat cutoff" bug — `isekai_api:repeat`
     tiles by folding the query coordinate into a canonical [0, period) cell
     before evaluating `f`; a sphere whose true (untiled) extent crosses a
     multiple of the period gets its far side evaluated against a wildly
     different effective coordinate and is clipped by a flat seam plane.
  2. Pairwise surface-to-surface separation (3D, `mode: xyz`) is at least
     MIN_SURFACE_GAP for every pair of spheres, searching periodic images over
     a shift window wide enough that further shifts can only be farther away.
  3. Every sphere's Y extent stays within the overworld's generated Y range.
  4. Visibility: every sphere has at least MIN_VISIBLE_NEIGHBORS other spheres
     (any periodic image) within MAX_VISIBLE_GAP surface-to-surface distance,
     so the sky is never empty at typical render distance.
  5. XZ column non-occlusion: `isekai_api:world_floor` walks a column straight
     down and only ever returns the *topmost* object's floor, so any pair of
     spheres whose XZ (not 3D) disks overlap will have the lower sphere's core
     ore silently fail to place in the overlap footprint — the 3D spacing
     check in (2) does not catch this because two spheres can be far apart in
     Y (and thus pass the 3D surface-gap check) while still overlapping when
     projected onto the XZ plane. This check requires every pair (including a
     sphere against its own periodic images, so a future radius bump past
     period/2 gets caught too) to have non-negative XZ surface-to-surface gap
     over the same periodic-image search as (2).

  6. Spawn safety: with an empty `spawn_target` vanilla starts its spawn search at
     BlockPos.ZERO and scans chunk (0,0), so the untiled home sphere must cover that
     chunk's corner columns with real thickness. Cross-reads the overworld
     noise_settings so a future non-empty `spawn_target` also trips this check.

Effective radius, not nominal radius
------------------------------------
The relief term is a world-seed-driven `minecraft:noise` added to the whole field, so
the surface of a sphere sits where `r + relief(x,y,z) - distance = 0` — the sphere's
*effective* radius is `r + relief`, not `r`. Every check below therefore runs on the
worst-case effective radius for the direction of risk it guards, read out of
`density_function/planet_relief.json`:

  * `r_out = r + max(0, relief_max)` — used by every check that guards against a
    sphere being *too big* (tile-boundary clipping, pairwise spacing, Y range,
    XZ occlusion).
  * `r_in  = r + min(0, relief_min)` — used by every check that guards against a
    sphere being *too small* (spawn column thickness, neighbour visibility).

The relief file clamps the noise to a fixed interval before scaling, so `relief_min`
and `relief_max` are exact structural bounds, not an estimate of what the noise
typically produces. With the shipped inward-only relief (`relief_max == 0`) the
"too big" checks reduce to exactly the pre-relief geometry check — that is the point:
the relief cannot push a sphere past a tile seam or into a neighbour, by construction.

Exit 0 with "OK" when every check passes; exit 1 listing every violation
otherwise.

Run: python tools/verify_cosmos_layout.py
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

FIELD_PATH = (
    REPO_ROOT
    / "src/main/resources/data/planet_world/worldgen/density_function/planet_field.json"
)

NOISE_SETTINGS_PATH = (
    REPO_ROOT
    / "src/main/resources/data/minecraft/worldgen/noise_settings/overworld.json"
)

RELIEF_PATH = (
    REPO_ROOT
    / "src/main/resources/data/planet_world/worldgen/density_function/planet_relief.json"
)

# The only density function planet_field.json is allowed to pull in by reference.
RELIEF_REF = "planet_world:planet_relief"

BOUNDARY_MARGIN = 8.0
MIN_SURFACE_GAP = 100.0
Y_MIN, Y_MAX = -64.0, 320.0
MAX_VISIBLE_GAP = 220.0
MIN_VISIBLE_NEIGHBORS = 2
SHIFT_RANGE = range(-2, 3)  # -2..2 inclusive; safe for period >> 2*(max radius + gap)

# Spawn-safety (check 6). Vanilla 1.21.1 Climate.Sampler#findSpawnPosition returns
# BlockPos.ZERO whenever noise_settings' spawn_target is empty, so
# MinecraftServer#setInitialSpawn starts its search at chunk (0,0) and
# PlayerRespawnLogic#getSpawnPosInChunk scans blocks x,z in 0..15 of that chunk. The
# player therefore lands on the home planet only while the untiled sphere still covers
# that 16x16 footprint with real terrain.
SPAWN_CHUNK_CORNERS = ((0, 0), (0, 15), (15, 0), (15, 15))
MIN_SPAWN_COLUMN_THICKNESS = 24.0  # blocks; also the depth the shallow ore bands need


@dataclass
class Sphere:
    name: str
    cx: float
    cy: float
    cz: float
    r: float  # nominal radius, straight out of the JSON constant
    period: float | None  # None => not tiled (e.g. the home planet)
    # Worst-case effective radii once the relief term is added. Defaulted so callers
    # that only care about the nominal geometry (probe_planet_ores.py) keep working.
    r_out: float = 0.0  # largest the sphere can ever get
    r_in: float = 0.0  # smallest the sphere can ever get

    def __post_init__(self) -> None:
        if self.r_out == 0.0:
            self.r_out = self.r
        if self.r_in == 0.0:
            self.r_in = self.r


def load_relief_bounds(
    relief_path: pathlib.Path = RELIEF_PATH, required: bool = False
) -> tuple[float, float]:
    """Exact (min, max) of the relief term added to the whole planet field.

    Only recognises the one shape the field is allowed to use:
    ``mul(depth, clamp(input, min, max))``. The clamp is what makes the bounds exact
    rather than an estimate — `minecraft:noise` alone has no tight analytic bound, so
    a relief built without a clamp would leave every downstream check guessing. Any
    other shape raises instead of silently returning a wrong (and too generous) bound.

    ``required`` is set by the caller once it knows whether the field tree actually
    references the relief. A referenced-but-missing file has to be an error here: the
    checks would otherwise pass on a clean (0, 0) relief while the game fails to load
    the datapack at all.
    """
    if not relief_path.exists():
        if required:
            raise ValueError(
                f"{FIELD_PATH.name} references {RELIEF_REF} but {relief_path} does not "
                "exist — the datapack will fail to resolve the density function"
            )
        return (0.0, 0.0)
    node = json.loads(relief_path.read_text(encoding="utf-8"))
    if node.get("type") != "minecraft:mul":
        raise ValueError(
            f"{relief_path.name}: expected top-level minecraft:mul carrying the single "
            f"depth constant, got {node.get('type')!r}"
        )
    depth = node.get("argument1")
    inner = node.get("argument2")
    if not isinstance(depth, (int, float)):
        raise ValueError(
            f"{relief_path.name}: argument1 must be the literal depth constant "
            "(the one number this file exists to expose), not a nested function"
        )
    if not isinstance(inner, dict) or inner.get("type") != "minecraft:clamp":
        raise ValueError(
            f"{relief_path.name}: argument2 must be a minecraft:clamp so the relief has "
            "hard bounds; without it the layout checks below have nothing to bound"
        )
    lo, hi = float(depth) * float(inner["min"]), float(depth) * float(inner["max"])
    return (min(lo, hi), max(lo, hi))


def _walk(node: dict | str, path: str, period: float | None, out: list[Sphere]) -> None:
    if isinstance(node, str):
        raise ValueError(
            f"{path}: density function reference {node!r} in a position this checker "
            "cannot resolve; only the relief term may be referenced, and only as a "
            "direct child of the root add()"
        )
    node_type = node["type"]
    if node_type == "isekai_api:max":
        _walk(node["a"], path + ".a", period, out)
        _walk(node["b"], path + ".b", period, out)
        return
    if node_type == "isekai_api:add" and (
        isinstance(node["a"], str) or isinstance(node["b"], str)
    ):
        # The relief wrapper: add(<sphere tree>, "planet_world:planet_relief"). Its
        # magnitude is read separately by load_relief_bounds() and folded into every
        # sphere's r_out / r_in, so here it is only validated and stepped over.
        for side in ("a", "b"):
            child = node[side]
            if isinstance(child, str):
                if child != RELIEF_REF:
                    raise ValueError(
                        f"{path}.{side}: unexpected density function reference "
                        f"{child!r} (only {RELIEF_REF!r} is accounted for by the "
                        "effective-radius checks below)"
                    )
            else:
                _walk(child, path + "." + side, period, out)
        return
    if node_type == "isekai_api:repeat":
        px, pz = node["period_x"], node["period_z"]
        if px != pz:
            raise ValueError(
                f"{path}: period_x != period_z ({px} vs {pz}), unsupported by this checker"
            )
        _walk(node["f"], path + ".f", px, out)
        return
    if node_type == "isekai_api:add":
        const_node, other = node["a"], node["b"]
        if const_node["type"] != "isekai_api:constant":
            const_node, other = node["b"], node["a"]
        if (
            const_node["type"] != "isekai_api:constant"
            or other["type"] != "isekai_api:negate"
        ):
            raise ValueError(
                f"{path}: expected add(constant, negate(distance)) sphere shape"
            )
        dist_node = other["f"]
        if dist_node["type"] != "isekai_api:distance":
            raise ValueError(f"{path}: expected negate(distance(...))")
        out.append(
            Sphere(
                name=path,
                cx=dist_node["ref_x"],
                cy=dist_node["ref_y"],
                cz=dist_node["ref_z"],
                r=const_node["value"],
                period=period,
            )
        )
        return
    raise ValueError(f"{path}: unrecognized node type {node_type!r}")


def field_references_relief(data: dict) -> bool:
    """True when the field tree pulls the relief in by reference. `_walk` only tolerates
    that reference as a direct child of the root add(), so this is the only place it can
    legally appear."""
    return data.get("type") == "isekai_api:add" and RELIEF_REF in (
        data.get("a"),
        data.get("b"),
    )


def load_spheres(field_path: pathlib.Path = FIELD_PATH) -> list[Sphere]:
    data = json.loads(field_path.read_text(encoding="utf-8"))
    spheres: list[Sphere] = []
    _walk(data, "root", None, spheres)
    relief_min, relief_max = load_relief_bounds(required=field_references_relief(data))
    for s in spheres:
        s.r_out = s.r + max(0.0, relief_max)
        s.r_in = s.r + min(0.0, relief_min)
    return spheres


def min_surface_gap(a: Sphere, b: Sphere, radius: str = "r_out") -> float:
    """Minimum surface-to-surface distance between a and b over all periodic
    images (a body with period=None only exists at its literal position).

    ``radius`` picks which effective radius the surfaces sit at: ``r_out`` (the
    default) for "are they too close", ``r_in`` for "are they still close enough
    to see each other"."""
    ra, rb = getattr(a, radius), getattr(b, radius)
    shifts_a = SHIFT_RANGE if a.period is not None else (0,)
    shifts_b = SHIFT_RANGE if b.period is not None else (0,)
    best = math.inf
    for sxa, sza in itertools.product(shifts_a, shifts_a):
        ax = a.cx + sxa * (a.period or 0.0)
        az = a.cz + sza * (a.period or 0.0)
        for sxb, szb in itertools.product(shifts_b, shifts_b):
            if a is b and (sxa, sza) == (sxb, szb):
                continue
            bx = b.cx + sxb * (b.period or 0.0)
            bz = b.cz + szb * (b.period or 0.0)
            d = math.sqrt((ax - bx) ** 2 + (a.cy - b.cy) ** 2 + (az - bz) ** 2)
            gap = d - ra - rb
            if gap < best:
                best = gap
    return best


def min_xz_surface_gap(a: Sphere, b: Sphere) -> float:
    """Minimum XZ-only surface-to-surface distance between a and b over all
    periodic images (including a vs its own other periodic images when
    a is b). This is the quantity `isekai_api:world_floor` column occlusion
    actually cares about: two spheres can be arbitrarily far apart in Y (and
    so pass min_surface_gap's 3D check) while still overlapping when the
    column is collapsed onto the XZ plane, which silently drops the lower
    sphere's floor-anchored ore for every column in the overlap.

    Runs on ``r_out`` — the overlap footprint is set by the widest the spheres can
    ever get, so a relief that can bulge outward has to be paid for here."""
    shifts_a = SHIFT_RANGE if a.period is not None else (0,)
    shifts_b = SHIFT_RANGE if b.period is not None else (0,)
    best = math.inf
    for sxa, sza in itertools.product(shifts_a, shifts_a):
        ax = a.cx + sxa * (a.period or 0.0)
        az = a.cz + sza * (a.period or 0.0)
        for sxb, szb in itertools.product(shifts_b, shifts_b):
            if a is b and (sxa, sza) == (sxb, szb):
                continue
            bx = b.cx + sxb * (b.period or 0.0)
            bz = b.cz + szb * (b.period or 0.0)
            d = math.hypot(ax - bx, az - bz)
            gap = d - a.r_out - b.r_out
            if gap < best:
                best = gap
    return best


def check_boundary(sphere: Sphere) -> str | None:
    if sphere.period is None:
        return None
    lo = sphere.r_out + BOUNDARY_MARGIN
    hi = sphere.period - (sphere.r_out + BOUNDARY_MARGIN)
    cxm = sphere.cx % sphere.period
    czm = sphere.cz % sphere.period
    if not (lo <= cxm <= hi):
        return f"{sphere.name}: cx%period={cxm:.2f} outside safe [{lo:.2f},{hi:.2f}]"
    if not (lo <= czm <= hi):
        return f"{sphere.name}: cz%period={czm:.2f} outside safe [{lo:.2f},{hi:.2f}]"
    return None


def check_y_extent(sphere: Sphere) -> str | None:
    lo, hi = sphere.cy - sphere.r_out, sphere.cy + sphere.r_out
    if lo < Y_MIN or hi > Y_MAX:
        return f"{sphere.name}: y-extent [{lo:.1f},{hi:.1f}] breaks bounds [{Y_MIN},{Y_MAX}]"
    return None


def check_spawn_safety(spheres: list[Sphere]) -> list[str]:
    """Check 6: the player's initial spawn lands on the home planet.

    Three things have to hold together, so they are checked together:

    1. ``spawn_target`` in the overworld noise_settings is empty, which is what makes
       vanilla hand back ``BlockPos.ZERO`` and search chunk (0,0) first.
    2. Exactly one sphere is untiled — that is the home planet, the only one pinned to
       absolute coordinates rather than repeated every ``period`` blocks.
    3. Every corner column of chunk (0,0) is inside the home sphere with at least
       ``MIN_SPAWN_COLUMN_THICKNESS`` blocks of rock under it, so the spawn search finds
       standable ground rather than a rim sliver (or void, which drops the player at
       y=sea_level with nothing under them).

    Point 3 runs on ``r_in`` — the *smallest* the home planet can get once the relief
    term carves into it. A relief deep enough to thin the spawn columns below the floor
    fails here even though it leaves every other check untouched, which is the whole
    reason the two effective radii are tracked separately.
    """
    violations: list[str] = []

    settings = json.loads(NOISE_SETTINGS_PATH.read_text(encoding="utf-8"))
    spawn_target = settings.get("spawn_target")
    if spawn_target:
        violations.append(
            f"spawn_target is non-empty ({len(spawn_target)} entry/entries); the spawn "
            "search no longer starts at BlockPos.ZERO, so landing on the home planet is "
            "no longer implied by this file"
        )

    home = [s for s in spheres if s.period is None]
    if len(home) != 1:
        violations.append(
            f"expected exactly 1 untiled (home) sphere, found {len(home)}"
        )
        return violations

    h = home[0]
    for cx, cz in SPAWN_CHUNK_CORNERS:
        d2 = (cx - h.cx) ** 2 + (cz - h.cz) ** 2
        if d2 >= h.r_in * h.r_in:
            violations.append(
                f"{h.name}: chunk-(0,0) column ({cx},{cz}) is outside the home sphere "
                f"(xz dist={math.sqrt(d2):.2f} >= r_in={h.r_in:.1f})"
            )
            continue
        thickness = 2.0 * math.sqrt(h.r_in * h.r_in - d2)
        if thickness < MIN_SPAWN_COLUMN_THICKNESS:
            violations.append(
                f"{h.name}: chunk-(0,0) column ({cx},{cz}) is only {thickness:.1f} blocks "
                f"thick (need >= {MIN_SPAWN_COLUMN_THICKNESS})"
            )
    return violations


def main() -> int:
    spheres = load_spheres()
    violations: list[str] = []

    for s in spheres:
        msg = check_boundary(s)
        if msg:
            violations.append("BOUNDARY: " + msg)
        msg = check_y_extent(s)
        if msg:
            violations.append("Y-RANGE: " + msg)

    for a, b in itertools.combinations(spheres, 2):
        gap = min_surface_gap(a, b)
        if gap < MIN_SURFACE_GAP:
            violations.append(
                f"SPACING: {a.name} - {b.name}: gap={gap:.2f} < {MIN_SURFACE_GAP}"
            )

    for a, b in itertools.combinations(spheres, 2):
        xz_gap = min_xz_surface_gap(a, b)
        if xz_gap < 0:
            violations.append(
                f"XZ-OCCLUSION: {a.name} - {b.name}: xz_gap={xz_gap:.2f} < 0 "
                "(world_floor column occlusion — lower sphere loses its core ore "
                "in the overlap footprint)"
            )
    for a in spheres:
        if a.period is None:
            continue
        xz_gap = min_xz_surface_gap(a, a)
        if xz_gap < 0:
            violations.append(
                f"XZ-OCCLUSION: {a.name} vs its own periodic image: "
                f"xz_gap={xz_gap:.2f} < 0 (radius >= period/2)"
            )

    for a in spheres:
        # r_in: if the relief carves both surfaces inward they drift apart, so the
        # worst case for "can you still see a neighbour" is the smallest both can be.
        n = sum(
            1
            for b in spheres
            if b is not a and min_surface_gap(a, b, "r_in") <= MAX_VISIBLE_GAP
        )
        if n < MIN_VISIBLE_NEIGHBORS:
            violations.append(
                f"VISIBILITY: {a.name}: only {n} neighbor(s) within {MAX_VISIBLE_GAP} "
                f"(need >= {MIN_VISIBLE_NEIGHBORS})"
            )

    violations.extend("SPAWN: " + m for m in check_spawn_safety(spheres))

    relief_min, relief_max = load_relief_bounds(required=True)
    print(f"Parsed {len(spheres)} sphere(s) from {FIELD_PATH}:")
    for s in spheres:
        tiled = f"period={s.period:.0f}" if s.period is not None else "not tiled"
        print(
            f"  {s.name:40s} r={s.r:5.1f} "
            f"eff=[{s.r_in:5.1f},{s.r_out:5.1f}] "
            f"center=({s.cx:.1f}, {s.cy:.1f}, {s.cz:.1f}) {tiled}"
        )
    print(
        f"Relief term {RELIEF_REF}: [{relief_min:+.2f}, {relief_max:+.2f}] blocks "
        f"(from {RELIEF_PATH.name})"
    )
    print()

    if violations:
        print(f"FAILED: {len(violations)} violation(s)")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(
        "OK: no boundary, spacing, Y-range, visibility, XZ-occlusion, or spawn violations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

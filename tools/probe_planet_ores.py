#!/usr/bin/env python3
"""RCON block-probe: can a player survive the home planet, and does every planet carry ore?

Companion to `verify_cosmos_layout.py`, which only checks the *layout* JSON. This
one talks to a running headless dedicated server and counts real generated blocks.

Two gates, both run by default:

  HOME  — the starting-resource gate. Counts wood, stone, coal, iron and water
          inside the whole home sphere, and finds the topmost solid block of the
          spawn column (0,·,0). This is what "the player does not get stuck at the
          starting point" means in practice.
  LAYERS — the per-planet ore gate inherited from isekai-api: three Y slabs per
          sphere, one `/fill … replace <ore>` per ore.

Planet coordinates are read from `planet_field.json` via
`verify_cosmos_layout.load_spheres()` — never hand-copied, so the probe cannot
drift away from the layout it is measuring.

Counting works by `/fill <box> minecraft:air replace <filter>` and reading the
"Successfully filled N block(s)" reply, so **it destroys what it counts**. Run it
against a throwaway probe world only, and only once per world.

Server setup (dedicated server, not runClient — a client exposes no RCON):

    run/server.properties      enable-rcon=true, rcon.port=25599,
                               rcon.password=planetworld, level-type=minecraft:normal
    run/eula.txt               eula=true
    ./gradlew runServer        (from mod-063-planet-world)

Usage:
    python tools/probe_planet_ores.py [--host 127.0.0.1] [--port 25599]
                                      [--password planetworld] [--half-width 8]
                                      [--home-only | --layers-only] [--per-biome]

Exit 0 when the home planet carries all five starting resources and every probed
planet has coal or copper in its upper slab, iron somewhere, and diamond or lapis
in its lower slab; exit 1 listing the failures.
"""

from __future__ import annotations

import argparse
import re
import select
import socket
import struct
import sys
import time

from verify_cosmos_layout import Sphere, load_spheres

# logical ore -> the block ids it can generate as (plain + deepslate variant)
ORE_BLOCKS = {
    "coal": ("minecraft:coal_ore", "minecraft:deepslate_coal_ore"),
    "copper": ("minecraft:copper_ore", "minecraft:deepslate_copper_ore"),
    "iron": ("minecraft:iron_ore", "minecraft:deepslate_iron_ore"),
    "redstone": ("minecraft:redstone_ore", "minecraft:deepslate_redstone_ore"),
    "gold": ("minecraft:gold_ore", "minecraft:deepslate_gold_ore"),
    "lapis": ("minecraft:lapis_ore", "minecraft:deepslate_lapis_ore"),
    "diamond": ("minecraft:diamond_ore", "minecraft:deepslate_diamond_ore"),
    "emerald": ("minecraft:emerald_ore", "minecraft:deepslate_emerald_ore"),
}

SLABS = ("lower", "middle", "upper")
FILLED_RE = re.compile(r"Successfully filled (\d+)")

# The five resources KURONAMI333's scope declares the home planet must guarantee, each as
# the /fill block predicate that counts it and the minimum count that clears the gate.
# `wood` is a worldgen tree count, so a handful of logs is already enough to bootstrap;
# `water` only has to yield one bucket.
HOME_RESOURCES = {
    "wood": ("#minecraft:logs", 16),
    "stone": ("minecraft:stone", 1000),
    "coal": ("#minecraft:coal_ores", 24),
    "iron": ("#minecraft:iron_ores", 24),
    "water": ("minecraft:water", 1),
}

# /fill refuses any box larger than this many blocks.
FILL_VOLUME_LIMIT = 32768


class Rcon:
    """Minimal synchronous RCON client (no third-party dependency)."""

    def __init__(self, host: str, port: int, password: str) -> None:
        self.sock = socket.create_connection((host, port), timeout=30)
        self.req_id = 0
        if self._send(3, password) is None:
            raise SystemExit("RCON auth failed")

    def _send(self, packet_type: int, body: str) -> str | None:
        self.req_id += 1
        sent_id = self.req_id
        payload = (
            struct.pack("<ii", sent_id, packet_type)
            + body.encode("utf-8")
            + b"\x00\x00"
        )
        self.sock.sendall(struct.pack("<i", len(payload)) + payload)
        length = struct.unpack("<i", self._recv_exact(4))[0]
        resp_id, _ = struct.unpack("<ii", self._recv_exact(8))
        text = self._recv_exact(length - 8)[:-2].decode("utf-8", "replace")
        if resp_id != sent_id:
            return None  # auth failure is signalled by resp_id == -1
        return text

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise SystemExit("RCON connection closed")
            buf += chunk
        return buf

    def cmd(self, command: str) -> str:
        return self._send(2, command) or ""

    def close(self) -> None:
        self.sock.close()


def wait_loaded(rcon: Rcon, x: int, y: int, z: int, timeout: float = 180.0) -> bool:
    """Block until the chunk containing (x,y,z) has generated.

    A naive probe against an ungenerated chunk silently reports neither branch,
    which historically read as a false 'solid'. Force-load first, then poll.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "The time is" in rcon.cmd(
            f"execute if loaded {x} {y} {z} run time query gametime"
        ):
            return True
        time.sleep(2.0)
    return False


def solid_volume(s: Sphere, w: int, y0: int, y1: int) -> int:
    """Blocks of the sphere inside the (2w+1)^2 probe window between y0 and y1.

    The denominator for the density ceiling. Computed geometrically from the
    sphere rather than measured, because the ore fills have already removed the
    blocks by the time we would want to count them.
    """
    total = 0
    for y in range(y0, y1 + 1):
        dy = y - s.cy
        rr = s.r * s.r - dy * dy
        if rr <= 0:
            continue
        for dx in range(-w, w + 1):
            for dz in range(-w, w + 1):
                if dx * dx + dz * dz <= rr:
                    total += 1
    return total


def count_filter(
    rcon: Rcon, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, block_filter: str
) -> int:
    """Count (and destroy) every block matching `block_filter` in the box.

    Split into Y sub-slabs so no single /fill exceeds the vanilla volume limit.
    """
    layer = (x1 - x0 + 1) * (z1 - z0 + 1)
    step = max(1, FILL_VOLUME_LIMIT // layer)
    total = 0
    y = y0
    while y <= y1:
        top = min(y + step - 1, y1)
        reply = rcon.cmd(
            f"fill {x0} {y} {z0} {x1} {top} {z1} minecraft:air replace {block_filter}"
        )
        m = FILLED_RE.search(reply)
        total += int(m.group(1)) if m else 0
        y = top + 1
    return total


def spawn_column_top(
    rcon: Rcon, x: int, z: int, y_hi: int, y_lo: int
) -> tuple[int, str] | None:
    """Topmost non-air block of the column, as (y, "grass"|"other").

    Mirrors what vanilla's spawn search walks down to: MinecraftServer#setInitialSpawn
    hands PlayerRespawnLogic#getSpawnPosInChunk chunk (0,0) whenever spawn_target is
    empty, and that scans down for the first block with a full upward face.
    """
    for y in range(y_hi, y_lo - 1, -1):
        reply = rcon.cmd(
            f"execute unless block {x} {y} {z} minecraft:air run time query gametime"
        )
        if "The time is" not in reply:
            continue
        grass = rcon.cmd(
            f"execute if block {x} {y} {z} minecraft:grass_block run time query gametime"
        )
        return y, ("grass" if "The time is" in grass else "other")
    return None


def probe_home(
    rcon: Rcon, home: Sphere
) -> tuple[dict[str, int], tuple[int, str] | None]:
    """Starting-resource gate over the whole home sphere (destructive)."""
    r = int(home.r)
    x, z = int(home.cx), int(home.cz)
    y_lo, y_hi = int(home.cy - home.r), int(home.cy + home.r)
    x0, x1, z0, z1 = x - r, x + r, z - r, z + r

    rcon.cmd(f"forceload add {x0} {z0} {x1} {z1}")
    if not wait_loaded(rcon, x, int(home.cy), z):
        raise SystemExit("home: chunks never finished generating")

    # Read the spawn column before the fills empty it out.
    top = spawn_column_top(rcon, x, z, y_hi + 2, y_lo)

    counts = {
        name: count_filter(rcon, x0, y_lo, z0, x1, y_hi, z1, block_filter)
        for name, (block_filter, _) in HOME_RESOURCES.items()
    }
    rcon.cmd(f"forceload remove {x0} {z0} {x1} {z1}")
    return counts, top


def slab_bounds(s: Sphere) -> list[tuple[str, int, int]]:
    lo, hi = int(s.cy - s.r), int(s.cy + s.r)
    step = (hi - lo) / 3.0
    edges = [lo, int(lo + step), int(lo + 2 * step), hi]
    return [(SLABS[i], edges[i], edges[i + 1]) for i in range(3)]


def probe(rcon: Rcon, s: Sphere, half: int) -> dict[str, dict[str, int]]:
    w = max(2, min(half, int(s.r) - 4))
    x, z = int(s.cx), int(s.cz)
    rcon.cmd(f"forceload add {x - w} {z - w} {x + w} {z + w}")
    if not wait_loaded(rcon, x, int(s.cy), z):
        raise SystemExit(f"{s.name}: chunks never finished generating")

    counts: dict[str, dict[str, int]] = {}
    for slab, y0, y1 in slab_bounds(s):
        counts[slab] = {"_solid": solid_volume(s, w, y0, y1)}
        for ore, blocks in ORE_BLOCKS.items():
            total = 0
            for block in blocks:
                reply = rcon.cmd(
                    f"fill {x - w} {y0} {z - w} {x + w} {y1} {z + w} "
                    f"minecraft:air replace {block}"
                )
                m = FILLED_RE.search(reply)
                total += int(m.group(1)) if m else 0
            counts[slab][ore] = total
    rcon.cmd(f"forceload remove {x - w} {z - w} {x + w} {z + w}")
    return counts


BIOMES = [
    "planet_verdant",
    "planet_ember",
    "planet_frost",
    "planet_stone",
    "planet_desert",
    "planet_mushroom",
    "planet_jungle",
    "planet_volcanic",
    "planet_crystal",
    "planet_dead",
]
MAX_ORE_FRACTION = 0.05  # any single ore filling >5% of a slab is a compression bug
AT_ZERO_RE = re.compile(r"\(0 blocks? away\)")
SHIFTS = range(-2, 3)


def biome_at(rcon: Rcon, x: int, y: int, z: int) -> str | None:
    """Which planet biome owns (x,y,z)? `locate biome` reporting "0 blocks away"
    means the probe point IS that biome (no chunk loading required)."""
    for b in BIOMES:
        reply = rcon.cmd(
            f"execute positioned {x} {y} {z} run locate biome planet_world:{b}"
        )
        if AT_ZERO_RE.search(reply):
            return b
    return None


def find_planet_per_biome(rcon: Rcon, spheres: list[Sphere]) -> dict[str, Sphere]:
    """One concrete planet instance per biome, searched over periodic images.

    Needed because the biome hash is an XZ noise independent of the planet grid,
    so a biome is not tied to any one layer — the only way to prove that e.g.
    planet_volcanic (blackstone filler) really produces ore is to find a
    volcanic planet and dig it.
    """
    found: dict[str, Sphere] = {}
    for s in spheres:
        shifts = SHIFTS if s.period is not None else (0,)
        for sx in shifts:
            for sz in shifts:
                if len(found) == len(BIOMES):
                    return found
                x = int(s.cx + sx * (s.period or 0.0))
                z = int(s.cz + sz * (s.period or 0.0))
                b = biome_at(rcon, x, int(s.cy), z)
                if b and b not in found:
                    found[b] = Sphere(f"{b}@{x},{int(s.cy)},{z}", x, s.cy, z, s.r, None)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25599)
    ap.add_argument("--password", default="planetworld")
    ap.add_argument("--half-width", type=int, default=8)
    ap.add_argument(
        "--per-biome",
        action="store_true",
        help="probe one planet of each of the 10 biomes instead of "
        "one planet of each layer",
    )
    ap.add_argument(
        "--home-only",
        action="store_true",
        help="run only the home-planet starting-resource gate",
    )
    ap.add_argument(
        "--layers-only",
        action="store_true",
        help="run only the per-planet ore gate",
    )
    args = ap.parse_args()

    spheres = load_spheres()
    rcon = Rcon(args.host, args.port, args.password)
    violations: list[str] = []

    if not args.layers_only:
        home_spheres = [s for s in spheres if s.period is None]
        if len(home_spheres) != 1:
            violations.append(
                f"expected exactly 1 untiled (home) sphere, found {len(home_spheres)}"
            )
        else:
            home = home_spheres[0]
            counts, top = probe_home(rcon, home)
            print(
                f"\nHOME  r={home.r:.0f}  center="
                f"({home.cx:.0f}, {home.cy:.0f}, {home.cz:.0f})"
            )
            for name, (block_filter, floor) in HOME_RESOURCES.items():
                got = counts[name]
                mark = "ok " if got >= floor else "NG "
                print(f"  {mark}{name:<6} {got:>8}  (need >= {floor}, {block_filter})")
                if got < floor:
                    violations.append(
                        f"home: {name} count {got} < {floor} ({block_filter})"
                    )
            if top is None:
                print("  NG spawn column (0,z=0) is empty from top to bottom")
                violations.append(
                    "home: spawn column (0,0) has no solid block — the player spawns "
                    "in the void"
                )
            else:
                y, kind = top
                print(f"  ok spawn column (0,0) top = y{y} ({kind})")
                if kind != "grass":
                    violations.append(
                        f"home: spawn column (0,0) top block at y{y} is not grass_block"
                    )

    if args.home_only:
        rcon.close()
        print()
        if violations:
            print(f"FAILED: {len(violations)} violation(s)")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("OK: the home planet carries all five starting resources")
        return 0

    if args.per_biome:
        found = find_planet_per_biome(rcon, spheres)
        missing = [b for b in BIOMES if b not in found]
        if missing:
            violations.append(f"no planet instance found for: {', '.join(missing)}")
        spheres = [found[b] for b in BIOMES if b in found]

    for s in spheres:
        counts = probe(rcon, s, args.half_width)
        print(f"\n{s.name}  r={s.r:.0f}  center=({s.cx:.0f}, {s.cy:.0f}, {s.cz:.0f})")
        header = "  slab    " + "".join(f"{o:>10}" for o in ORE_BLOCKS)
        print(header)
        for slab in SLABS:
            vol = counts[slab]["_solid"]
            pct = (
                f"  ({100 * sum(counts[slab][o] for o in ORE_BLOCKS) / vol:.1f}% of {vol})"
                if vol
                else ""
            )
            print(
                f"  {slab:<8}"
                + "".join(f"{counts[slab][o]:>10}" for o in ORE_BLOCKS)
                + pct
            )

        for slab in SLABS:
            vol = counts[slab]["_solid"]
            for ore in ORE_BLOCKS:
                if vol and counts[slab][ore] / vol > MAX_ORE_FRACTION:
                    violations.append(
                        f"{s.name}: {ore} is {100 * counts[slab][ore] / vol:.1f}% of the "
                        f"{slab} slab (ceiling {100 * MAX_ORE_FRACTION:.0f}%)"
                    )

        upper_shallow = counts["upper"]["coal"] + counts["upper"]["copper"]
        iron_any = sum(counts[sl]["iron"] for sl in SLABS)
        lower_deep = counts["lower"]["diamond"] + counts["lower"]["lapis"]
        if upper_shallow == 0:
            violations.append(f"{s.name}: no coal/copper in the upper slab")
        if iron_any == 0:
            violations.append(f"{s.name}: no iron anywhere")
        if lower_deep == 0:
            violations.append(f"{s.name}: no diamond/lapis in the lower slab")

    rcon.close()
    print()
    if violations:
        print(f"FAILED: {len(violations)} violation(s)")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(
        "OK: the home planet carries all five starting resources, and every planet has "
        "shallow ore up top, iron, and deep ore at the core"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

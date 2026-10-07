#!/usr/bin/env python3
"""Print the Minecraft API signatures the Fabric mod depends on.

The mod is compiled against Mojang's official mappings, which are not available
outside a Gradle/Loom build. When the CI build fails on a renamed method there is
no way to read the compiler output from a restricted environment, so this helper
answers the question directly: it finds the Minecraft jar Loom resolved into the
Gradle cache and dumps (with ``javap``) the members the recorder calls.

Usage::

    python3 scripts/mc_api_probe.py            # uses ~/.gradle/caches
    python3 scripts/mc_api_probe.py --out probe.log

Everything goes to stdout so it can be teed into a file that gets published as a
check summary. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# (class, regex of the members to show, "does this class exist at all?")
LOOKUPS: list[tuple[str, str]] = [
    ("net.minecraft.client.Options", r"key"),
    ("net.minecraft.client.KeyMapping", r"isDown|consumeClick|getKey"),
    ("net.minecraft.world.entity.player.Inventory", r"selected|slot"),
    ("net.minecraft.world.entity.LivingEntity", r"food|health|getYRot|getXRot"),
    ("net.minecraft.world.entity.player.Player", r"inventory|sprint|shift|food|deltaMovement|onGround"),
    ("net.minecraft.world.entity.Entity", r"position|getDeltaMovement|getType|blockPosition|onGround"),
    ("net.minecraft.resources.ResourceKey", r"."),
    ("net.minecraft.world.level.Level", r"dimension|getBlockState|getEntities"),
    ("net.minecraft.core.Registry", r"getKey"),
    ("net.minecraft.core.registries.BuiltInRegistries", r"BLOCK|ENTITY_TYPE"),
    ("net.minecraft.core.registries.Registries", r"DIMENSION"),
    ("net.minecraft.core.BlockPos", r"offset|relative"),
    ("net.minecraft.world.phys.Vec3", r"distanceTo|length|subtract"),
    ("net.minecraft.world.level.block.state.BlockState", r"getBlock|isAir"),
    ("net.minecraft.client.player.LocalPlayer", r"."),
]

# Renames we are not sure about: report which of these exist.
EXISTENCE = [
    "net.minecraft.resources.Identifier",
    "net.minecraft.resources.ResourceLocation",
    "net.minecraft.Identifier",
    "net.minecraft.world.entity.monster.Monster",
    "net.minecraft.world.entity.animal.Animal",
    "net.minecraft.world.menu.QuickCraftButton",
]


def find_minecraft_jar(cache: Path) -> Path | None:
    candidates: list[Path] = []
    for path in cache.rglob("*.jar"):
        name = path.name.lower()
        if "source" in name or "sources" in name:
            continue
        if "minecraft" not in name and "mojang" not in name and "merged" not in name:
            continue
        candidates.append(path)
    # Newest first: the last build that resolved the jar is the interesting one.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if javap(path, "net.minecraft.client.Options"):
            return path
    return None


def javap(jar: Path, class_name: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed javap invocation
            ["javap", "-classpath", str(jar), class_name],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - CI only
        return f"(javap failed: {exc})"
    if result.returncode != 0:
        return ""
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache",
        default=os.environ.get("GRADLE_USER_HOME") or str(Path.home() / ".gradle" / "caches"),
        help="Gradle cache directory to search",
    )
    parser.add_argument("--jar", default="", help="use this Minecraft jar instead of searching the cache")
    args = parser.parse_args(argv)

    out: list[str] = []
    jar = Path(args.jar) if args.jar else find_minecraft_jar(Path(args.cache))
    if jar is None:
        out.append("could not find a mapped Minecraft jar in the Gradle cache")
        jars = sorted({str(p) for p in Path(args.cache).rglob("*.jar")})[:40]
        out.append("first jars seen:")
        out.extend(jars)
    else:
        out.append(f"minecraft jar: {jar}")
        out.append("")
        for class_name, pattern in LOOKUPS:
            out.append(f"== {class_name}  (/{pattern}/)")
            text = javap(jar, class_name)
            if not text:
                out.append("   <class not found under this name>")
            else:
                import re

                kept = [line for line in text.splitlines() if re.search(pattern, line, re.IGNORECASE)]
                out.extend(kept or ["   <no member matched>"])
            out.append("")
        out.append("== class existence")
        for class_name in EXISTENCE:
            text = javap(jar, class_name)
            out.append(f"   {'FOUND   ' if text else 'MISSING '} {class_name}")

    text = "\n".join(out)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CI helper
    sys.exit(main())

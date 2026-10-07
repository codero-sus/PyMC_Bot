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
    ("net.minecraft.resources.Identifier", r"fromNamespaceAndPath|getNamespace|getPath|withDefaultNamespace"),
    ("net.minecraft.client.Minecraft", r"options|getInstance|level|player"),
    ("net.minecraft.world.entity.EquipmentSlot", r"HEAD|CHEST|LEGS|FEET"),
    ("net.minecraft.world.entity.LivingEntity", r"getItemBySlot|getMainHandItem|getArmorSlots|getHealth"),
    ("net.minecraft.world.entity.player.Inventory", r"getItem|getSelected|armor"),
    ("net.minecraft.world.entity.item.ItemEntity", r"getItem"),
    ("net.minecraft.world.item.ItemStack", r"isEmpty|getCount|getItem"),
    ("net.minecraft.core.registries.BuiltInRegistries", r"ITEM|BLOCK|ENTITY_TYPE"),
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


def find_minecraft_jar(cache: Path) -> tuple[Path | None, list[Path]]:
    """The newest jar that really contains the mapped classes, plus everything tried."""
    candidates: list[Path] = []
    for path in cache.rglob("*.jar"):
        name = path.name.lower()
        if "source" in name or "sources" in name:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    tried: list[Path] = []
    for path in candidates:
        # A jar counts as "the game" when the classes we call are really in it: Loom
        # caches both the mapped game and small helper jars with similar names.
        if javap(path, "net.minecraft.client.Options") and javap(path, "net.minecraft.world.entity.Entity"):
            return path, tried
        tried.append(path)
        if len(tried) >= 60:
            break
    return None, tried


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
    tried: list[Path] = []
    if args.jar:
        jar: Path | None = Path(args.jar)
    else:
        jar, tried = find_minecraft_jar(Path(args.cache))
    if jar is None:
        out.append(
            f"could not find a mapped Minecraft jar (checked {len(tried)} jar(s) in "
            f"{args.cache}); the classes may be named differently in this version"
        )
        out.append("newest jars seen:")
        out.extend(str(path) for path in tried[:40])
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

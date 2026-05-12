#!/usr/bin/env python3
from pathlib import Path
import xml.etree.ElementTree as ET

SRC = Path(
    "/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/"
    "b2z1_isaacsim_mesh_axis_fixed.urdf"
)

# fallback
if not SRC.exists():
    SRC = Path(
        "/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/"
        "b2z1.urdf"
    )

DST = Path(
    "/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/"
    "b2z1_isaacsim_feet_only_collision.urdf"
)

KEEP_COLLISION_LINKS = {
    "base_link",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
}

tree = ET.parse(SRC)
root = tree.getroot()

removed = []
kept = []

for link in root.findall("link"):
    name = link.attrib.get("name", "")
    collisions = list(link.findall("collision"))

    if not collisions:
        continue

    if name in KEEP_COLLISION_LINKS:
        kept.append(name)
        continue

    for collision in collisions:
        link.remove(collision)
    removed.append(name)

DST.parent.mkdir(parents=True, exist_ok=True)
tree.write(DST, encoding="utf-8", xml_declaration=True)

print("SRC:", SRC)
print("DST:", DST)
print("Kept collision links:", sorted(set(kept)))
print("Removed collision from links:", sorted(set(removed)))

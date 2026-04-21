import copy
import math
import os
import xml.etree.ElementTree as ET


def _sanitize_token(token):
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in token)


def _format_float(value):
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _write_text_if_changed(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _build_obj(vertices, faces):
    lines = []
    for x, y, z in vertices:
        lines.append(f"v {_format_float(x)} {_format_float(y)} {_format_float(z)}")
    for face in faces:
        lines.append("f " + " ".join(str(index) for index in face))
    return "\n".join(lines) + "\n"


def _create_box_obj(size_xyz):
    hx, hy, hz = (0.5 * float(value) for value in size_xyz)
    vertices = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    return _build_obj(vertices, faces)


def _create_cylinder_obj(length, radius, segments=64):
    half_length = 0.5 * float(length)
    radius = float(radius)
    vertices = [(0.0, 0.0, half_length), (0.0, 0.0, -half_length)]
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        vertices.append((x, y, half_length))
        vertices.append((x, y, -half_length))

    faces = []
    top_center = 1
    bottom_center = 2
    for i in range(segments):
        next_i = (i + 1) % segments
        top_i = 3 + 2 * i
        bottom_i = top_i + 1
        top_next = 3 + 2 * next_i
        bottom_next = top_next + 1
        faces.append((top_center, top_next, top_i))
        faces.append((bottom_center, bottom_i, bottom_next))
        faces.append((top_i, top_next, bottom_next))
        faces.append((top_i, bottom_next, bottom_i))
    return _build_obj(vertices, faces)


def _create_uv_sphere_obj(radius, longitude_segments=64, latitude_segments=32):
    radius = float(radius)
    vertices = [(0.0, 0.0, radius)]
    for lat in range(1, latitude_segments):
        phi = math.pi * lat / latitude_segments
        z = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)
        for lon in range(longitude_segments):
            theta = 2.0 * math.pi * lon / longitude_segments
            x = ring_radius * math.cos(theta)
            y = ring_radius * math.sin(theta)
            vertices.append((x, y, z))
    vertices.append((0.0, 0.0, -radius))

    faces = []
    north_pole = 1
    south_pole = len(vertices)

    def ring_vertex(lat_idx, lon_idx):
        lon_idx %= longitude_segments
        return 2 + (lat_idx - 1) * longitude_segments + lon_idx

    for lon in range(longitude_segments):
        faces.append((north_pole, ring_vertex(1, lon + 1), ring_vertex(1, lon)))

    for lat in range(1, latitude_segments - 1):
        for lon in range(longitude_segments):
            current = ring_vertex(lat, lon)
            next_lon = ring_vertex(lat, lon + 1)
            below = ring_vertex(lat + 1, lon)
            below_next = ring_vertex(lat + 1, lon + 1)
            faces.append((current, next_lon, below_next))
            faces.append((current, below_next, below))

    last_ring = latitude_segments - 1
    for lon in range(longitude_segments):
        faces.append((south_pole, ring_vertex(last_ring, lon), ring_vertex(last_ring, lon + 1)))

    return _build_obj(vertices, faces)


def _create_capsule_obj(length, radius, longitude_segments=64, latitude_segments=16):
    length = float(length)
    radius = float(radius)
    half_length = 0.5 * length
    vertices = []

    def append_ring(z_center, phi_start, phi_end):
        for lat in range(latitude_segments + 1):
            phi = phi_start + (phi_end - phi_start) * lat / latitude_segments
            ring_radius = radius * math.sin(phi)
            z = z_center + radius * math.cos(phi)
            for lon in range(longitude_segments):
                theta = 2.0 * math.pi * lon / longitude_segments
                x = ring_radius * math.cos(theta)
                y = ring_radius * math.sin(theta)
                vertices.append((x, y, z))

    append_ring(half_length, 0.0, math.pi / 2.0)
    append_ring(-half_length, math.pi / 2.0, math.pi)

    top_pole = len(vertices) + 1
    vertices.append((0.0, 0.0, half_length + radius))
    bottom_pole = len(vertices) + 1
    vertices.append((0.0, 0.0, -half_length - radius))

    faces = []

    def ring_vertex(ring_idx, lon_idx):
        lon_idx %= longitude_segments
        return 1 + ring_idx * longitude_segments + lon_idx

    total_rings = 2 * (latitude_segments + 1)
    top_equator_ring = latitude_segments
    bottom_equator_ring = latitude_segments + 1

    for lon in range(longitude_segments):
        faces.append((top_pole, ring_vertex(0, lon + 1), ring_vertex(0, lon)))

    for ring in range(total_rings - 1):
        for lon in range(longitude_segments):
            current = ring_vertex(ring, lon)
            next_lon = ring_vertex(ring, lon + 1)
            below = ring_vertex(ring + 1, lon)
            below_next = ring_vertex(ring + 1, lon + 1)
            if ring != top_equator_ring:
                faces.append((current, next_lon, below_next))
                faces.append((current, below_next, below))
            else:
                faces.append((current, next_lon, below))
                faces.append((next_lon, below_next, below))

    for lon in range(longitude_segments):
        faces.append((bottom_pole, ring_vertex(total_rings - 1, lon), ring_vertex(total_rings - 1, lon + 1)))

    return _build_obj(vertices, faces)


def _find_geometry_child(geometry):
    for tag in ("box", "cylinder", "sphere", "mesh"):
        child = geometry.find(tag)
        if child is not None:
            return tag, child
    return None, None


def _build_visual_from_collision(collision, mesh_dir, output_dir, link_name, collision_index, use_capsule_for_cylinders):
    geometry = collision.find("geometry")
    if geometry is None:
        return None

    geometry_tag, geometry_child = _find_geometry_child(geometry)
    if geometry_tag is None:
        return None

    visual = ET.Element("visual")
    origin = collision.find("origin")
    if origin is not None:
        visual.append(copy.deepcopy(origin))

    visual_geometry = ET.SubElement(visual, "geometry")

    if geometry_tag == "mesh":
        visual_geometry.append(copy.deepcopy(geometry_child))
    else:
        safe_link_name = _sanitize_token(link_name or f"link_{collision_index}")
        mesh_filename = f"{safe_link_name}_collision_{collision_index}.obj"
        mesh_path = os.path.join(mesh_dir, mesh_filename)
        if geometry_tag == "box":
            size_xyz = geometry_child.attrib["size"].split()
            mesh_contents = _create_box_obj(size_xyz)
        elif geometry_tag == "cylinder":
            length = geometry_child.attrib["length"]
            radius = geometry_child.attrib["radius"]
            if use_capsule_for_cylinders:
                mesh_contents = _create_capsule_obj(length, radius)
            else:
                mesh_contents = _create_cylinder_obj(length, radius)
        else:
            radius = geometry_child.attrib["radius"]
            mesh_contents = _create_uv_sphere_obj(radius)
        _write_text_if_changed(mesh_path, mesh_contents)
        mesh_rel_path = os.path.relpath(mesh_path, output_dir).replace(os.sep, "/")
        ET.SubElement(visual_geometry, "mesh", filename=mesh_rel_path)

    material = ET.SubElement(visual, "material", name="collision_visual_material")
    ET.SubElement(material, "color", rgba="0.72 0.72 0.72 1.0")
    return visual


def ensure_collision_visual_urdf(source_urdf_path, use_capsule_for_cylinders=False):
    source_urdf_path = os.path.abspath(source_urdf_path)
    source_dir = os.path.dirname(source_urdf_path)
    source_stem, source_ext = os.path.splitext(os.path.basename(source_urdf_path))
    output_path = os.path.join(source_dir, f"{source_stem}_collision_visual{source_ext}")
    mesh_dir = os.path.join(source_dir, f"{source_stem}_collision_visual_meshes")

    tree = ET.parse(source_urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        existing_visuals = list(link.findall("visual"))
        for visual in existing_visuals:
            link.remove(visual)

        first_collision_index = next(
            (idx for idx, child in enumerate(list(link)) if child.tag == "collision"),
            len(link),
        )
        visuals_to_insert = []
        link_name = link.attrib.get("name", source_stem)
        for collision_index, collision in enumerate(collisions):
            visual = _build_visual_from_collision(
                collision,
                mesh_dir,
                source_dir,
                link_name,
                collision_index,
                use_capsule_for_cylinders,
            )
            if visual is not None:
                visuals_to_insert.append(visual)

        for offset, visual in enumerate(visuals_to_insert):
            link.insert(first_collision_index + offset, visual)

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    os.makedirs(source_dir, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path

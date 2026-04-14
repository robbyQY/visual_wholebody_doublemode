import os
import re
import math


VALID_MOUNT_DEGREES = (0, 90, 180, 270)
SOURCE_URDF_REL_PATH = os.path.join("resources", "robots", "b1z1", "urdf", "b1z1.urdf")
GENERATED_URDF_DIR_REL_PATH = os.path.join("resources", "robots", "b1z1", "urdf", "generated")


def _is_relative_resource_path(path):
    return not (
        os.path.isabs(path)
        or "://" in path
        or path.startswith("package://")
    )


def _rebase_urdf_resource_paths(line, source_dir, output_dir):
    def replace(match):
        original_path = match.group(1)
        if not _is_relative_resource_path(original_path):
            return match.group(0)

        source_abs_path = os.path.normpath(os.path.join(source_dir, original_path))
        rebased_path = os.path.relpath(source_abs_path, output_dir).replace(os.sep, "/")
        return match.group(0).replace(original_path, rebased_path, 1)

    return re.sub(r'filename="([^"]+)"', replace, line)


def normalize_mount_deg(mount_deg):
    mount_deg = int(round(float(mount_deg))) % 360
    if mount_deg not in VALID_MOUNT_DEGREES:
        raise ValueError(f"Unsupported B1Z1 mount_deg={mount_deg}. Expected one of {VALID_MOUNT_DEGREES}.")
    return mount_deg


def mount_deg_to_rad(mount_deg):
    return math.radians(normalize_mount_deg(mount_deg))


def get_generated_mount_urdf_rel_path(mount_deg):
    mount_deg = normalize_mount_deg(mount_deg)
    return os.path.join(GENERATED_URDF_DIR_REL_PATH, f"b1z1_mount_{mount_deg}.urdf")


def ensure_b1z1_mount_urdf(root_dir, mount_deg):
    mount_deg = normalize_mount_deg(mount_deg)
    source_path = os.path.join(root_dir, SOURCE_URDF_REL_PATH)
    output_rel_path = get_generated_mount_urdf_rel_path(mount_deg)
    output_path = os.path.join(root_dir, output_rel_path)
    source_dir = os.path.dirname(source_path)
    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_base_static_joint = False
    replaced_origin = False
    yaw_rad = mount_deg_to_rad(mount_deg)
    rewritten_lines = []

    for line in lines:
        line = _rebase_urdf_resource_paths(line, source_dir, output_dir)

        if '<joint name="base_static_joint"' in line:
            in_base_static_joint = True
            rewritten_lines.append(line)
            continue

        if in_base_static_joint and "<origin" in line:
            indent = re.match(r"\s*", line).group(0)
            numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            xyz_tokens = numeric_tokens[-3:] if len(numeric_tokens) >= 3 else ["0.3", "0", "0.09"]
            xyz_string = " ".join(f"{float(token):g}" for token in xyz_tokens)
            rewritten_lines.append(f'{indent}<origin rpy="0 0 {yaw_rad:.16g}" xyz="{xyz_string}" />\n')
            replaced_origin = True
            continue

        if in_base_static_joint and "</joint>" in line:
            in_base_static_joint = False

        rewritten_lines.append(line)

    if not replaced_origin:
        raise RuntimeError(f"Failed to rewrite base_static_joint origin in {source_path}")

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(rewritten_lines)

    return output_rel_path

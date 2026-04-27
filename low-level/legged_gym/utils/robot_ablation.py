import copy
import os
import xml.etree.ElementTree as ET

from .b1z1_mount import mount_deg_to_rad, normalize_mount_deg, normalize_mount_xyz


B1Z1_B2Z1_ROBOT_ABLATION_CHOICES = (
    "legs",
    "trunk",
    "arm",
    "mass",
    "inertial",
    "structure",
    "legs-mass",
    "legs-inertial",
    "legs-structure",
    "trunk-mass",
    "trunk-inertial",
    "trunk-structure",
    "arm-mass",
    "arm-inertial",
    "arm-structure",
)

_ROBOT_ABLATION_ALIASES = {
    "none": None,
}

_ROBOT_ABLATION_COMPONENTS = ("legs", "trunk", "arm")
_ROBOT_ABLATION_ASPECTS = ("mass", "inertial", "structure")

_LINK_MAP_B1_TO_B2 = {
    "trunk": "base_link",
    "imu_link": "imu_link",
    "FR_hip": "FR_hip",
    "FR_thigh": "FR_thigh",
    "FR_calf": "FR_calf",
    "FR_foot": "FR_foot",
    "FL_hip": "FL_hip",
    "FL_thigh": "FL_thigh",
    "FL_calf": "FL_calf",
    "FL_foot": "FL_foot",
    "RR_hip": "RR_hip",
    "RR_thigh": "RR_thigh",
    "RR_calf": "RR_calf",
    "RR_foot": "RR_foot",
    "RL_hip": "RL_hip",
    "RL_thigh": "RL_thigh",
    "RL_calf": "RL_calf",
    "RL_foot": "RL_foot",
    "link00": "link00",
    "link01": "link01",
    "link02": "link02",
    "link03": "link03",
    "link04": "link04",
    "link05": "link05",
    "link06": "link06",
    "gripperStator": "gripperStator",
    "gripperMover": "gripperMover",
    "ee_gripper_link": "gripper_link",
}

_JOINT_MAP_B1_TO_B2 = {
    "imu_joint": "joint_imu",
    "FR_hip_joint": "FR_hip_joint",
    "FR_thigh_joint": "FR_thigh_joint",
    "FR_calf_joint": "FR_calf_joint",
    "FR_foot_fixed": "FR_foot_joint",
    "FL_hip_joint": "FL_hip_joint",
    "FL_thigh_joint": "FL_thigh_joint",
    "FL_calf_joint": "FL_calf_joint",
    "FL_foot_fixed": "FL_foot_joint",
    "RR_hip_joint": "RR_hip_joint",
    "RR_thigh_joint": "RR_thigh_joint",
    "RR_calf_joint": "RR_calf_joint",
    "RR_foot_fixed": "RR_foot_joint",
    "RL_hip_joint": "RL_hip_joint",
    "RL_thigh_joint": "RL_thigh_joint",
    "RL_calf_joint": "RL_calf_joint",
    "RL_foot_fixed": "RL_foot_joint",
    "base_static_joint": "z1_mount_joint",
    "z1_waist": "joint1",
    "z1_shoulder": "joint2",
    "z1_elbow": "joint3",
    "z1_wrist_angle": "joint4",
    "z1_forearm_roll": "joint5",
    "z1_wrist_rotate": "joint6",
    "z1_gripperStator": "jointGripperStator",
    "z1_jointGripper": "jointGripper",
    "ee_gripper": "ee_gripper",
}

_COMPONENT_LINKS_B1 = {
    "trunk": (
        "trunk",
        "imu_link",
    ),
    "legs": (
        "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
        "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
        "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
        "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    ),
    "arm": (
        "link00",
        "link01",
        "link02",
        "link03",
        "link04",
        "link05",
        "link06",
        "gripperStator",
        "gripperMover",
        "ee_gripper_link",
    ),
}

_COMPONENT_JOINTS_B1 = {
    "trunk": (
        "imu_joint",
        "FR_hip_joint",
        "FL_hip_joint",
        "RR_hip_joint",
        "RL_hip_joint",
    ),
    "legs": (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_fixed",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_fixed",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_fixed",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_fixed",
    ),
    "arm": (
        "base_static_joint",
        "z1_waist",
        "z1_shoulder",
        "z1_elbow",
        "z1_wrist_angle",
        "z1_forearm_roll",
        "z1_wrist_rotate",
        "z1_gripperStator",
        "z1_jointGripper",
        "ee_gripper",
    ),
}


def _reverse_unique_mapping(mapping):
    return {value: key for key, value in mapping.items()}


def _remap_component_names(component_names, mapping):
    remapped = {}
    for component_name, names in component_names.items():
        remapped[component_name] = tuple(dict.fromkeys(mapping[name] for name in names))
    return remapped


_LINK_MAP_B2_TO_B1 = _reverse_unique_mapping(_LINK_MAP_B1_TO_B2)
_JOINT_MAP_B2_TO_B1 = _reverse_unique_mapping(_JOINT_MAP_B1_TO_B2)
_COMPONENT_LINKS_B2 = _remap_component_names(_COMPONENT_LINKS_B1, _LINK_MAP_B1_TO_B2)
_COMPONENT_JOINTS_B2 = _remap_component_names(_COMPONENT_JOINTS_B1, _JOINT_MAP_B1_TO_B2)

_ROBOT_ABLATION_SPECS = {
    "b1z1": {
        "other_robot": "b2z1",
        "source_urdf_rel_path": os.path.join("resources", "robots", "b1z1", "urdf", "b1z1.urdf"),
        "generated_urdf_dir_rel_path": os.path.join("resources", "robots", "b1z1", "urdf", "generated", "ablations"),
        "generated_filename_prefix": "b1z1_ablation",
        "mount_joint_name": "base_static_joint",
        "component_links": _COMPONENT_LINKS_B1,
        "component_joints": _COMPONENT_JOINTS_B1,
        "target_to_other_link_map": _LINK_MAP_B1_TO_B2,
        "target_to_other_joint_map": _JOINT_MAP_B1_TO_B2,
    },
    "b2z1": {
        "other_robot": "b1z1",
        "source_urdf_rel_path": os.path.join("resources", "robots", "b2z1", "urdf", "b2z1.urdf"),
        "generated_urdf_dir_rel_path": os.path.join("resources", "robots", "b2z1", "urdf", "generated", "ablations"),
        "generated_filename_prefix": "b2z1_ablation",
        "mount_joint_name": "z1_mount_joint",
        "component_links": _COMPONENT_LINKS_B2,
        "component_joints": _COMPONENT_JOINTS_B2,
        "target_to_other_link_map": _LINK_MAP_B2_TO_B1,
        "target_to_other_joint_map": _JOINT_MAP_B2_TO_B1,
    },
}


def canonicalize_b1z1_b2z1_robot_ablation(robot_ablation):
    if robot_ablation is None:
        return None
    if isinstance(robot_ablation, (list, tuple)):
        raw_tokens = robot_ablation
    else:
        normalized = str(robot_ablation).strip().lower()
        if not normalized:
            return None
        raw_tokens = normalized.replace(",", "+").split("+")

    canonical = []
    for raw_token in raw_tokens:
        normalized = str(raw_token).strip().lower()
        if not normalized:
            continue
        normalized = normalized.replace("_", "-").replace(".", "-").replace(":", "-")
        normalized = _ROBOT_ABLATION_ALIASES.get(normalized, normalized)
        if normalized is None:
            continue
        if normalized not in B1Z1_B2Z1_ROBOT_ABLATION_CHOICES:
            supported = ", ".join(B1Z1_B2Z1_ROBOT_ABLATION_CHOICES)
            raise ValueError(
                f"Unsupported robot_ablation={robot_ablation!r}. "
                f"Supported values: {supported}, none. Combine multiple values with ',' or '+'."
            )
        canonical.append(normalized)

    if not canonical:
        return None
    unique = set(canonical)
    return tuple(choice for choice in B1Z1_B2Z1_ROBOT_ABLATION_CHOICES if choice in unique)


def get_b1z1_b2z1_robot_ablation_checkpoint_value(robot_ablation):
    canonical = canonicalize_b1z1_b2z1_robot_ablation(robot_ablation)
    return "+".join(canonical) if canonical is not None else "none"


def normalize_leg_collision_scale(leg_collision_scale):
    if leg_collision_scale is None or str(leg_collision_scale).strip() == "":
        return 1.0
    scale = float(leg_collision_scale)
    if scale <= 0.0:
        raise ValueError(f"leg_collision_scale must be positive, got: {leg_collision_scale!r}")
    return scale


def _format_mount_xyz_token(value):
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _format_scale_token(value):
    text = f"{normalize_leg_collision_scale(value):.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _indent_xml(elem, level=0):
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _is_relative_resource_path(path):
    return not (
        os.path.isabs(path)
        or "://" in path
        or path.startswith("package://")
    )


def _rebase_resource_filenames(element, source_dir, output_dir):
    for node in element.iter():
        filename = node.attrib.get("filename")
        if not filename or not _is_relative_resource_path(filename):
            continue
        source_abs_path = os.path.normpath(os.path.join(source_dir, filename))
        rebased = os.path.relpath(source_abs_path, output_dir).replace(os.sep, "/")
        node.set("filename", rebased)


def _load_urdf_root(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Historical typo in the checked-in B1 URDF; keep the loader permissive.
    text = text.replace('xyz="0.3 0 0.09>>"', 'xyz="0.3 0 0.09"')
    return ET.fromstring(text)


def _find_named_elements(root, tag_name):
    return {
        element.attrib["name"]: element
        for element in root.findall(tag_name)
        if "name" in element.attrib
    }


def _replace_tag_group(target_element, source_element, tag_name, source_dir, output_dir):
    for existing in list(target_element):
        if existing.tag == tag_name:
            target_element.remove(existing)
    for source_child in source_element.findall(tag_name):
        cloned = copy.deepcopy(source_child)
        _rebase_resource_filenames(cloned, source_dir, output_dir)
        target_element.append(cloned)


def _replace_inertial(target_link, source_link, source_dir, output_dir):
    for existing in list(target_link):
        if existing.tag == "inertial":
            target_link.remove(existing)
    source_inertial = source_link.find("inertial")
    if source_inertial is not None:
        cloned = copy.deepcopy(source_inertial)
        _rebase_resource_filenames(cloned, source_dir, output_dir)
        target_link.append(cloned)


def _replace_mass_only(target_link, source_link):
    target_mass = target_link.find("./inertial/mass")
    source_mass = source_link.find("./inertial/mass")
    if target_mass is None or source_mass is None:
        return
    target_mass.set("value", source_mass.attrib["value"])


def _replace_joint_structure(target_joint, source_joint):
    for tag_name in ("origin", "axis", "limit", "dynamics"):
        for existing in list(target_joint):
            if existing.tag == tag_name:
                target_joint.remove(existing)
        source_child = source_joint.find(tag_name)
        if source_child is not None:
            target_joint.append(copy.deepcopy(source_child))


def _scale_float_list_attribute(node, attribute_name, scale):
    raw_value = node.attrib.get(attribute_name)
    if raw_value is None:
        return
    scaled_values = [f"{float(token) * scale:.12g}" for token in raw_value.split()]
    node.set(attribute_name, " ".join(scaled_values))


def _scale_collision_geometry(geometry_node, scale):
    box = geometry_node.find("box")
    if box is not None:
        _scale_float_list_attribute(box, "size", scale)
    sphere = geometry_node.find("sphere")
    if sphere is not None and "radius" in sphere.attrib:
        sphere.set("radius", f"{float(sphere.attrib['radius']) * scale:.12g}")
    cylinder = geometry_node.find("cylinder")
    if cylinder is not None:
        if "radius" in cylinder.attrib:
            cylinder.set("radius", f"{float(cylinder.attrib['radius']) * scale:.12g}")
        if "length" in cylinder.attrib:
            cylinder.set("length", f"{float(cylinder.attrib['length']) * scale:.12g}")
    capsule = geometry_node.find("capsule")
    if capsule is not None:
        if "radius" in capsule.attrib:
            capsule.set("radius", f"{float(capsule.attrib['radius']) * scale:.12g}")
        if "length" in capsule.attrib:
            capsule.set("length", f"{float(capsule.attrib['length']) * scale:.12g}")


def _scale_link_collision_geometry(root, link_names, scale):
    target_links = _find_named_elements(root, "link")
    for link_name in link_names:
        if not (link_name.endswith("_thigh") or link_name.endswith("_calf")):
            continue

        link = target_links[link_name]
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is not None:
                _scale_collision_geometry(geometry, scale)


def _set_mount_joint_origin(root, mount_joint_name, mount_deg, mount_xyz):
    mount_joint = _find_named_elements(root, "joint").get(mount_joint_name)
    if mount_joint is None:
        raise RuntimeError(f'Failed to find mount joint "{mount_joint_name}" in ablation URDF.')
    origin = mount_joint.find("origin")
    if origin is None:
        origin = ET.SubElement(mount_joint, "origin")
    origin.set("rpy", f"0 0 {mount_deg_to_rad(mount_deg):.16g}")
    origin.set("xyz", " ".join(f"{value:g}" for value in mount_xyz))


def _collect_component_element_names(component_names, component_name_to_names):
    names = []
    for component_name in component_names:
        names.extend(component_name_to_names[component_name])
    return tuple(dict.fromkeys(names))


def _apply_structure_ablation(
    target_root,
    source_root,
    component_names,
    component_links,
    component_joints,
    target_to_other_link_map,
    target_to_other_joint_map,
    source_dir,
    output_dir,
):
    target_links = _find_named_elements(target_root, "link")
    source_links = _find_named_elements(source_root, "link")
    target_joints = _find_named_elements(target_root, "joint")
    source_joints = _find_named_elements(source_root, "joint")

    for target_link_name in _collect_component_element_names(component_names, component_links):
        source_link_name = target_to_other_link_map[target_link_name]
        _replace_tag_group(target_links[target_link_name], source_links[source_link_name], "visual", source_dir, output_dir)
        _replace_tag_group(target_links[target_link_name], source_links[source_link_name], "collision", source_dir, output_dir)

    for target_joint_name in _collect_component_element_names(component_names, component_joints):
        source_joint_name = target_to_other_joint_map[target_joint_name]
        _replace_joint_structure(target_joints[target_joint_name], source_joints[source_joint_name])


def _apply_inertial_ablation(
    target_root,
    source_root,
    component_names,
    component_links,
    target_to_other_link_map,
    source_dir,
    output_dir,
):
    target_links = _find_named_elements(target_root, "link")
    source_links = _find_named_elements(source_root, "link")

    for target_link_name in _collect_component_element_names(component_names, component_links):
        source_link_name = target_to_other_link_map[target_link_name]
        _replace_inertial(target_links[target_link_name], source_links[source_link_name], source_dir, output_dir)


def _apply_mass_only_ablation(target_root, source_root, component_names, component_links, target_to_other_link_map):
    target_links = _find_named_elements(target_root, "link")
    source_links = _find_named_elements(source_root, "link")

    for target_link_name in _collect_component_element_names(component_names, component_links):
        source_link_name = target_to_other_link_map[target_link_name]
        _replace_mass_only(target_links[target_link_name], source_links[source_link_name])


def _apply_robot_ablation_token(
    target_root,
    source_root,
    ablation_name,
    spec,
    other_source_dir,
    output_dir,
):
    if ablation_name in _ROBOT_ABLATION_COMPONENTS:
        component_names = (ablation_name,)
        _apply_structure_ablation(
            target_root,
            source_root,
            component_names,
            spec["component_links"],
            spec["component_joints"],
            spec["target_to_other_link_map"],
            spec["target_to_other_joint_map"],
            other_source_dir,
            output_dir,
        )
        _apply_inertial_ablation(
            target_root,
            source_root,
            component_names,
            spec["component_links"],
            spec["target_to_other_link_map"],
            other_source_dir,
            output_dir,
        )
        return

    if ablation_name in _ROBOT_ABLATION_ASPECTS:
        component_names = _ROBOT_ABLATION_COMPONENTS
        aspect_name = ablation_name
    else:
        try:
            component_name, aspect_name = ablation_name.split("-", 1)
        except ValueError as exc:
            raise ValueError(f"Unsupported robot_ablation={ablation_name!r}") from exc
        component_names = (component_name,)

    if aspect_name == "mass":
        _apply_mass_only_ablation(
            target_root,
            source_root,
            component_names,
            spec["component_links"],
            spec["target_to_other_link_map"],
        )
    elif aspect_name == "inertial":
        _apply_inertial_ablation(
            target_root,
            source_root,
            component_names,
            spec["component_links"],
            spec["target_to_other_link_map"],
            other_source_dir,
            output_dir,
        )
    elif aspect_name == "structure":
        _apply_structure_ablation(
            target_root,
            source_root,
            component_names,
            spec["component_links"],
            spec["component_joints"],
            spec["target_to_other_link_map"],
            spec["target_to_other_joint_map"],
            other_source_dir,
            output_dir,
        )
    else:
        raise ValueError(f"Unsupported robot_ablation={ablation_name!r}")


def _get_robot_ablation_spec(base_robot):
    try:
        return _ROBOT_ABLATION_SPECS[base_robot]
    except KeyError as exc:
        supported = ", ".join(sorted(_ROBOT_ABLATION_SPECS.keys()))
        raise ValueError(f"Unsupported base_robot={base_robot!r}. Supported values: {supported}.") from exc


def get_generated_robot_ablation_urdf_rel_path(base_robot, robot_ablation, mount_deg, mount_xyz, leg_collision_scale=1.0):
    spec = _get_robot_ablation_spec(base_robot)
    robot_ablation = canonicalize_b1z1_b2z1_robot_ablation(robot_ablation)
    leg_collision_scale = normalize_leg_collision_scale(leg_collision_scale)
    if robot_ablation is None and leg_collision_scale == 1.0:
        raise ValueError("robot_ablation or leg_collision_scale must request a generated URDF")
    mount_deg = normalize_mount_deg(mount_deg)
    mount_xyz = normalize_mount_xyz(mount_xyz)
    xyz_suffix = "_".join(
        f"{axis}{_format_mount_xyz_token(value)}"
        for axis, value in zip(("x", "y", "z"), mount_xyz)
    )
    ablation_suffix = get_b1z1_b2z1_robot_ablation_checkpoint_value(robot_ablation)
    scale_suffix = ""
    if leg_collision_scale != 1.0:
        scale_suffix = f"_legcol{_format_scale_token(leg_collision_scale)}"
    filename = f"{spec['generated_filename_prefix']}_{ablation_suffix}_{mount_deg}_{xyz_suffix}{scale_suffix}.urdf"
    return os.path.join(spec["generated_urdf_dir_rel_path"], filename)


def ensure_cross_robot_ablation_urdf(root_dir, base_robot, robot_ablation, mount_deg, mount_xyz, leg_collision_scale=1.0):
    spec = _get_robot_ablation_spec(base_robot)
    other_spec = _get_robot_ablation_spec(spec["other_robot"])
    robot_ablation = canonicalize_b1z1_b2z1_robot_ablation(robot_ablation)
    leg_collision_scale = normalize_leg_collision_scale(leg_collision_scale)
    if robot_ablation is None and leg_collision_scale == 1.0:
        raise ValueError("robot_ablation or leg_collision_scale must request a generated URDF")

    mount_deg = normalize_mount_deg(mount_deg)
    mount_xyz = normalize_mount_xyz(mount_xyz)

    target_source_path = os.path.join(root_dir, spec["source_urdf_rel_path"])
    other_source_path = os.path.join(root_dir, other_spec["source_urdf_rel_path"])
    output_rel_path = get_generated_robot_ablation_urdf_rel_path(
        base_robot,
        robot_ablation,
        mount_deg,
        mount_xyz,
        leg_collision_scale,
    )
    output_path = os.path.join(root_dir, output_rel_path)
    output_dir = os.path.dirname(output_path)
    target_source_dir = os.path.dirname(target_source_path)
    other_source_dir = os.path.dirname(other_source_path)

    os.makedirs(output_dir, exist_ok=True)

    target_root = _load_urdf_root(target_source_path)
    other_root = _load_urdf_root(other_source_path)

    _rebase_resource_filenames(target_root, target_source_dir, output_dir)

    for ablation_name in robot_ablation or ():
        _apply_robot_ablation_token(
            target_root,
            other_root,
            ablation_name,
            spec,
            other_source_dir,
            output_dir,
        )

    if leg_collision_scale != 1.0:
        _scale_link_collision_geometry(target_root, spec["component_links"]["legs"], leg_collision_scale)

    _set_mount_joint_origin(target_root, spec["mount_joint_name"], mount_deg, mount_xyz)
    target_root.set("name", f"{base_robot}_{get_b1z1_b2z1_robot_ablation_checkpoint_value(robot_ablation)}_legcol{_format_scale_token(leg_collision_scale)}")
    _indent_xml(target_root)
    tree = ET.ElementTree(target_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_rel_path


def ensure_b1z1_ablation_urdf(root_dir, robot_ablation, mount_deg, mount_xyz, leg_collision_scale=1.0):
    return ensure_cross_robot_ablation_urdf(
        root_dir,
        "b1z1",
        robot_ablation,
        mount_deg,
        mount_xyz,
        leg_collision_scale,
    )

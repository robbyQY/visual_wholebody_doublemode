from __future__ import annotations
import numpy as np
import torch
from legged_gym.utils.math import orientation_error
from legged_gym.utils.isaaclab_math import (
    quat_rotate_inverse,
    quat_apply,
    quat_from_euler_xyz,
    euler_from_quat,
    wrap_to_pi,
)
from legged_gym.envs.base.legged_robot_isaaclab import LeggedRobotIsaacLab
from .b2z1_isaaclab_config import B2Z1IsaacLabCfg


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for wxyz quaternions."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply for wxyz quaternions."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def sphere2cart(sphere: torch.Tensor) -> torch.Tensor:
    """Legacy ManipLoco sphere convention: [radius, pitch, yaw] -> [x, y, z]."""
    radius = sphere[..., 0]
    pitch = sphere[..., 1]
    yaw = sphere[..., 2]
    cp = torch.cos(pitch)
    return torch.stack(
        [
            radius * cp * torch.cos(yaw),
            radius * cp * torch.sin(yaw),
            radius * torch.sin(pitch),
        ],
        dim=-1,
    )


def cart2sphere(cart: torch.Tensor) -> torch.Tensor:
    """Legacy ManipLoco cartesian convention: [x, y, z] -> [radius, pitch, yaw]."""
    x, y, z = cart.unbind(-1)
    radius = torch.linalg.norm(cart, dim=-1).clamp_min(1e-8)
    pitch = torch.asin(torch.clamp(z / radius, -1.0, 1.0))
    yaw = torch.atan2(y, x)
    return torch.stack([radius, pitch, yaw], dim=-1)



class ManipLocoIsaacLab(LeggedRobotIsaacLab):
    """IsaacLab simulator-layer port for the uploaded ManipLoco task.

    This class keeps the old B2/Z1 action dimension and core simulator operations.
    The original IK/Jacobian/box-object reward stack is preserved in legacy files and
    should be migrated after the USD body names and Jacobian APIs are verified.
    """
    cfg: B2Z1IsaacLabCfg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_teleop_state()
        self._init_debug_draw()

    def _init_teleop_state(self):
        self.teleop_mode = bool(getattr(self.cfg.env, "teleop_mode", False)) if hasattr(self.cfg, "env") else False
        self.teleop_raw_commands = torch.zeros_like(self.commands)
        self.teleop_arm_control_mode = "ee"

        # Legacy teleop EE command buffers. In old Gym, curr_ee_goal_cart is in the
        # goal-local frame, while curr_ee_goal_cart_world is the IK target in world.
        self.teleop_raw_ee_goal_cart = self.curr_ee_goal_cart.clone()
        self.teleop_raw_ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.teleop_hold_actual_ee_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.teleop_arm_joint_pos_targets = self.dof_pos[:, self.arm_joint_ids].clone()
        self.gripper_pos_targets = self.default_dof_pos[:, self.gripper_joint_ids].clone()
        self.teleop_debug = True

        # World-frame IK target buffers. Do NOT initialize world target from local command.
        # They are synchronized to the real EE pose once body_state_w is available.
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_goal_orn_quat[:, 0] = 1.0
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self._teleop_ee_synced_once = False

        # Minimal legacy goal-frame constants from B2Z1RoughCfg. These are task math,
        # not simulator backend calls. Keep them here until the full cfg.goal_ee tree is ported.
        self.arm_induced_pitch = 0.38
        self.uses_goal_height_reference_mask = bool(getattr(self.cfg, "mixed_height_reference", False))

        self.ee_goal_center_offset = torch.tensor(
            [0.2, 0.0, 0.7], device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self.arm_base_offset = torch.tensor(
            [0.2, 0.0, 0.09], device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self.arm_waist_offset = self.arm_base_offset.clone()
        self.arm_waist_offset[:, 2] += 0.0585
        self.arm_shoulder_offset = self.arm_waist_offset.clone()
        self.arm_shoulder_offset[:, 2] += 0.045

        # Goal trajectory placeholders for legacy-compatible helpers.
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch.ones(self.num_envs, device=self.device)
        self.traj_total_timesteps = torch.ones(self.num_envs, device=self.device) * 1.0e9
        if not hasattr(self, "curr_ee_goal_sphere"):
            self.curr_ee_goal_sphere = cart2sphere(self.curr_ee_goal_cart)
        else:
            self.curr_ee_goal_sphere[:] = cart2sphere(self.curr_ee_goal_cart)

        self._update_base_yaw_quat()

    def _init_debug_draw(self):
        self._debug_draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as e:
                print(f"[debug_draw][warn] debug draw unavailable: {e}")
                self._debug_draw = None

    def _pre_physics_step(self, actions: torch.Tensor):
        # old ManipLoco zeroed arm action columns before converting policy->env.
        actions = actions.clone()
        if actions.shape[-1] > 12:
            actions[:, 12:] = 0.0

        # Simulator replacement for old refresh_rigid_body_state_tensor /
        # refresh_jacobian_tensors before IK target generation.
        self._refresh_ee_and_jacobian_for_ik()

        # Keep old ManipLoco EE-goal math. This updates:
        #   curr_ee_goal_cart_world: world-frame IK position target
        #   ee_goal_orn_quat:       world-frame IK orientation target
        self._update_curr_ee_goal()

        # Parent only handles action clipping/delay/target_pos for legs.
        super()._pre_physics_step(actions)

    def _update_effective_teleop_inputs(self):
        if not getattr(self, "teleop_mode", False):
            return

        self.commands[:] = self.teleop_raw_commands

        # Preserve old behavior: when joint-controlling the arm, keep EE target synced
        # to actual pose so switching back to EE mode does not jump.
        if self.teleop_arm_control_mode == "joint":
            if hasattr(self, "ee_pos") and hasattr(self, "ee_orn"):
                self._sync_teleop_ee_goal_to_current_pose()
            return

        self.curr_ee_goal_cart[:] = self.teleop_raw_ee_goal_cart
        self.ee_goal_orn_delta_rpy[:] = self.teleop_raw_ee_goal_orn_delta_rpy
        self.curr_ee_goal_sphere[:] = cart2sphere(self.curr_ee_goal_cart)

    def _toggle_teleop_arm_control_mode(self):
        next_mode = "joint" if self.teleop_arm_control_mode == "ee" else "ee"
        self._set_teleop_arm_control_mode(next_mode)

    def apply_teleop_key(self, key: str):
        """Legacy-compatible teleop key mapping (run_teleop.sh)."""
        k = key.lower()
        if k == "q":
            self.teleop_raw_commands[:, 0] = 0.0
        elif k == "w":
            self.teleop_raw_commands[:, 0] += 0.05
        elif k == "s":
            self.teleop_raw_commands[:, 0] -= 0.05
        elif k == "e":
            self.teleop_raw_commands[:, 2] = 0.0
        elif k == "a":
            self.teleop_raw_commands[:, 2] += 0.05
        elif k == "d":
            self.teleop_raw_commands[:, 2] -= 0.05
        elif k == "g":
            self._toggle_teleop_arm_control_mode()

        if self.teleop_arm_control_mode == "joint":
            delta = 0.05
            # NOTE: use 'v' instead of 'h' to avoid IsaacSim hotkey conflict.
            if k in "yvujikzxcmbn":
                mapping = {"y": (0, +delta), "v": (0, -delta), "u": (1, +delta), "j": (1, -delta),
                           "i": (2, +delta), "k": (2, -delta), "z": (3, +delta), "x": (3, -delta),
                           "c": (4, +delta), "m": (4, -delta), "b": (5, +delta), "n": (5, -delta)}
                idx, dv = mapping[k]
                self.teleop_arm_joint_pos_targets[:, idx] += dv
            elif k == "l":
                self.teleop_arm_joint_pos_targets[:] = self.default_dof_pos[:, self.arm_joint_ids]
        else:
            if k == "y":
                self.teleop_raw_ee_goal_cart[:, 0] += 0.05
            elif k == "v":
                self.teleop_raw_ee_goal_cart[:, 0] -= 0.05
            elif k == "u":
                self.teleop_raw_ee_goal_cart[:, 1] += 0.05
            elif k == "j":
                self.teleop_raw_ee_goal_cart[:, 1] -= 0.05
            elif k == "i":
                self.teleop_raw_ee_goal_cart[:, 2] += 0.05
            elif k == "k":
                self.teleop_raw_ee_goal_cart[:, 2] -= 0.05

        if k == "o":
            self.gripper_pos_targets += 0.05
        elif k == "p":
            self.gripper_pos_targets -= 0.05

        self._update_effective_teleop_inputs()
        print(
            f"[teleop][env] key={key} cmd={self.teleop_raw_commands[0, :3].detach().cpu().tolist()} "
            f"arm_mode={self.teleop_arm_control_mode} ee_goal={self.curr_ee_goal_cart[0].detach().cpu().tolist()}"
        )

    def _refresh_ee_and_jacobian_for_ik(self):
        """Refresh IK input tensors at step boundary (legacy-compatible timing)."""
        if not hasattr(self, "body_names_to_idx"):
            return
        if not hasattr(self, "gripper_body_name"):
            self.gripper_body_name = getattr(self.cfg, "gripper_body_name", "gripper_link")
        if self.gripper_body_name not in self.body_names_to_idx:
            return

        self._update_base_yaw_quat()
        self.gripper_idx = self.body_names_to_idx[self.gripper_body_name]

        # EE pose from current rigid-body state (world frame).
        if hasattr(self.robot.data, "body_state_w"):
            body_state_w = self.robot.data.body_state_w
            self.ee_pos = body_state_w[:, self.gripper_idx, :3]
            self.ee_orn = body_state_w[:, self.gripper_idx, 3:7]
            if hasattr(self.robot.data, "body_vel_w"):
                self.ee_vel = self.robot.data.body_vel_w[:, self.gripper_idx, :]
            else:
                self.ee_vel = torch.zeros(self.num_envs, 6, device=self.device)

            # First teleop frame: exactly mimic old Gym teleop EE sync.
            # This should make dpos≈0 and drot≈0 before any key is pressed.
            if getattr(self, "teleop_mode", False) and not getattr(self, "_teleop_ee_synced_once", False):
                self._sync_teleop_ee_goal_to_current_pose()
                self._teleop_ee_synced_once = True
                print("[teleop][sync] initialized EE goal from current EE pose")

        # Arm DOFs are non-contiguous in IsaacLab sim order: [8, 13, 14, 15, 16, 17].
        # Never use arm_dof_start_idx:arm_dof_end_idx for IsaacLab arm tensors.
        if hasattr(self, "arm_joint_ids") and self.arm_joint_ids.numel() > 0:
            self.arm_dof_start_idx = int(self.arm_joint_ids.min().item())
            self.arm_dof_end_idx = int(self.arm_joint_ids.max().item()) + 1

        self.ee_j_eef = None
        root_view = getattr(self.robot, "root_physx_view", None)
        if root_view is not None and hasattr(root_view, "get_jacobians"):
            try:
                jac = root_view.get_jacobians()
                if jac is not None and hasattr(self, "arm_joint_ids"):
                    # Use explicit arm IDs because IsaacLab sim joint order is not old Gym order.
                    # self.ee_j_eef = jac[:, self.gripper_idx, :6, self.arm_joint_ids]
                    floating_base_offset = jac.shape[-1] - self.num_dofs  # should be 6 for floating base
                    jac_arm_joint_ids = self.arm_joint_ids + floating_base_offset
                    self.ee_j_eef = jac[:, self.gripper_idx, :6, jac_arm_joint_ids]
            except Exception:
                self.ee_j_eef = None

        if int(getattr(self, "global_steps", 0)) % 20 == 0:
            print("[JAC DEBUG] jac full shape:", tuple(jac.shape))
            print("[JAC DEBUG] gripper_idx body_state:", int(self.gripper_idx))
            print("[JAC DEBUG] body_names:", self.body_names)

            for row in [self.gripper_idx - 2, self.gripper_idx - 1, self.gripper_idx, self.gripper_idx + 1]:
                if row < 0 or row >= jac.shape[1]:
                    continue
                J = jac[:, row, :6, self.arm_joint_ids]
                col_norm = torch.norm(J[0], dim=0)
                row_norm = torch.norm(J[0], dim=1)
                print(
                    f"[JAC DEBUG] row={row} "
                    f"col_norm={col_norm.detach().cpu().tolist()} "
                    f"row_norm={row_norm.detach().cpu().tolist()}"
                )


    @property
    def base_pos(self):
        return self.root_states[:, :3]

    def _update_base_yaw_quat(self):
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat(
            [torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)],
            dim=1,
        )
        zero = torch.zeros_like(base_yaw)
        self.base_yaw_quat = quat_from_euler_xyz(zero, zero, base_yaw)

    def _sync_teleop_arm_joint_targets_to_current_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self.teleop_arm_joint_pos_targets[env_ids] = self.dof_pos[env_ids][:, self.arm_joint_ids]

    def _sync_teleop_ee_goal_to_current_pose(self, env_ids=None):
        """Old Gym-compatible sync: make the EE target equal the actual EE pose."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        if not (hasattr(self, "ee_pos") and hasattr(self, "ee_orn")):
            return

        self._update_base_yaw_quat()
        goal_ref_origin = self.get_goal_reference_origin()[env_ids]
        goal_ref_quat = self.get_goal_reference_quat()[env_ids]
        goal_center_offset = self.get_goal_center_offset_local()[env_ids]
        ee_orn_normalized = self.ee_orn[env_ids] / torch.norm(
            self.ee_orn[env_ids], dim=-1, keepdim=True
        ).clamp_min(1e-8)

        ee_goal_local_with_center = quat_rotate_inverse(
            goal_ref_quat, self.ee_pos[env_ids] - goal_ref_origin
        )
        self.teleop_raw_ee_goal_cart[env_ids] = ee_goal_local_with_center - goal_center_offset
        self.curr_ee_goal_cart[env_ids] = self.teleop_raw_ee_goal_cart[env_ids]
        self.curr_ee_goal_sphere[env_ids] = cart2sphere(self.curr_ee_goal_cart[env_ids])

        local_ee_orn = quat_mul(quat_conjugate(goal_ref_quat), ee_orn_normalized)
        local_ee_orn_rpy = torch.stack(euler_from_quat(local_ee_orn), dim=-1)
        default_pitch = -self.curr_ee_goal_sphere[env_ids, 1] + self.arm_induced_pitch
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 0] = wrap_to_pi(local_ee_orn_rpy[:, 0] - np.pi / 2)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 1] = wrap_to_pi(local_ee_orn_rpy[:, 1] - default_pitch)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 2] = wrap_to_pi(
            local_ee_orn_rpy[:, 2] - self.curr_ee_goal_sphere[env_ids, 2]
        )
        self.ee_goal_orn_delta_rpy[env_ids] = self.teleop_raw_ee_goal_orn_delta_rpy[env_ids]
        self.teleop_hold_actual_ee_target[env_ids] = True

        self.curr_ee_goal_cart_world[env_ids] = self.ee_pos[env_ids]
        self.ee_goal_orn_quat[env_ids] = ee_orn_normalized
        self.ee_goal_orn_euler[env_ids] = torch.stack(
            euler_from_quat(self.ee_goal_orn_quat[env_ids]), dim=-1
        )

    def _set_teleop_arm_control_mode(self, mode):
        if mode == self.teleop_arm_control_mode:
            return
        if mode == "joint":
            self._sync_teleop_arm_joint_targets_to_current_pose()
        elif mode == "ee":
            self._sync_teleop_ee_goal_to_current_pose()
        else:
            raise ValueError(f"Unsupported teleop arm control mode: {mode}")
        self.teleop_arm_control_mode = mode
        print(f"[teleop] arm control mode: {mode}")

    def _update_curr_ee_goal(self):
        """Old ManipLoco EE target update, with simulator data supplied by IsaacLab."""
        if getattr(self, "teleop_mode", False):
            self._update_effective_teleop_inputs()
        else:
            # Non-teleop goal trajectory is not fully ported here; keep the current local goal.
            self.curr_ee_goal_sphere[:] = cart2sphere(self.curr_ee_goal_cart)

        goal_ref_quat = self.get_goal_reference_quat()
        self.curr_ee_goal_cart_world = self.transform_goal_local_to_world(
            self.get_goal_center_offset_local() + self.curr_ee_goal_cart
        )

        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.arm_induced_pitch
        local_goal_orn = quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + self.curr_ee_goal_sphere[:, 2],
        )
        self.ee_goal_orn_quat = quat_mul(goal_ref_quat, local_goal_orn)
        self.ee_goal_orn_euler = torch.stack(euler_from_quat(self.ee_goal_orn_quat), dim=-1)
        self.goal_timer += 1

    def _control_ik(self, dpose: torch.Tensor) -> torch.Tensor:
        """Damped least-squares IK in legacy style."""
        # if getattr(self, "ee_j_eef", None) is None:
        #     return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    def _get_arm_pos_targets(self) -> torch.Tensor:
        """Legacy-equivalent arm target generation.

        Same old Gym formula:
            dpose = [goal_pos_world - ee_pos, orientation_error(goal_quat, ee_quat)]
            arm_q_target = q_arm_now + damped_least_squares_IK(dpose)

        IsaacLab difference: arm joints are non-contiguous in sim order, so use arm_joint_ids.
        """
        if not all(hasattr(self, name) for name in ["curr_ee_goal_cart_world", "ee_pos", "ee_goal_orn_quat", "ee_orn", "dof_pos"]):
            return self.target_pos[:, self.arm_joint_ids]
        if getattr(self, "ee_j_eef", None) is None:
            return self.target_pos[:, self.arm_joint_ids]

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        ee_orn_norm = self.ee_orn / torch.norm(self.ee_orn, dim=-1, keepdim=True).clamp_min(1e-8)
        drot = orientation_error(self.ee_goal_orn_quat, ee_orn_norm)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        return self._control_ik(dpose) + self.dof_pos[:, self.arm_joint_ids]

    def get_goal_reference_quat(self):
        """Returns the goal-reference orientation in world coordinates."""
        self._update_base_yaw_quat()
        if not getattr(self, "uses_goal_height_reference_mask", False):
            return self.base_yaw_quat
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)

    def get_goal_reference_origin(self):
        """Returns the goal-reference origin in world coordinates."""
        invariant_origin = self.get_invariant_goal_reference_origin()
        if not getattr(self, "uses_goal_height_reference_mask", False):
            return invariant_origin
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_pos, invariant_origin)

    def get_invariant_goal_reference_origin(self, env_ids=None):
        if env_ids is None:
            root_xy = self.root_states[:, :2]
            num_envs = self.num_envs
        else:
            root_xy = self.root_states[env_ids, :2]
            num_envs = len(env_ids)
        return torch.cat(
            [root_xy, torch.zeros(num_envs, 1, device=self.device)],
            dim=1,
        )

    def get_goal_center_offset_local(self):
        """Returns the target-center offset in the goal local frame."""
        if not getattr(self, "uses_goal_height_reference_mask", False):
            return self.ee_goal_center_offset
        trunk_follow_anchor = getattr(getattr(getattr(self.cfg, "goal_ee", None), "sphere_center", None), "trunk_follow_anchor", "arm_waist")
        if trunk_follow_anchor == "arm_base":
            trunk_follow_center_offset = self.arm_base_offset
        elif trunk_follow_anchor == "arm_waist":
            trunk_follow_center_offset = self.arm_waist_offset
        elif trunk_follow_anchor == "arm_shoulder":
            trunk_follow_center_offset = self.arm_shoulder_offset
        else:
            raise ValueError(f"Unsupported trunk_follow_anchor: {trunk_follow_anchor}")
        return torch.where(
            self.goal_height_follow_mask.unsqueeze(1),
            trunk_follow_center_offset,
            self.ee_goal_center_offset,
        )

    def transform_goal_local_to_world(self, local_points):
        """Maps points from the goal local frame to world coordinates."""
        return self.get_goal_reference_origin() + quat_apply(self.get_goal_reference_quat(), local_points)

    def get_ee_goal_spherical_center(self):
        """Returns the cyan-sphere center in world coordinates."""
        return self.transform_goal_local_to_world(self.get_goal_center_offset_local())

    def _project_world_points_to_goal_sphere(self, env_ids, world_points):
        goal_ref_quat = self.get_goal_reference_quat()[env_ids]
        goal_ref_origin = self.get_goal_reference_origin()[env_ids]
        goal_center_offset = self.get_goal_center_offset_local()[env_ids]
        local_with_center = quat_rotate_inverse(goal_ref_quat, world_points - goal_ref_origin)
        return cart2sphere(local_with_center - goal_center_offset)

    def _get_reset_init_goal_world(self, env_ids):
        reset_init_cart = sphere2cart(self.reset_init_ee_sphere[env_ids])
        invariant_origin = self.get_invariant_goal_reference_origin(env_ids)
        invariant_center = self.ee_goal_center_offset[env_ids]
        return invariant_origin + quat_apply(self.base_yaw_quat[env_ids], invariant_center + reset_init_cart)

    def _get_arm_base_world_pos(self):
        """Returns the arm-base origin in world coordinates."""
        self._update_base_yaw_quat()
        arm_base_quat = self.base_yaw_quat
        if getattr(self, "uses_goal_height_reference_mask", False):
            arm_base_quat = torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)
        return self.base_pos + quat_apply(arm_base_quat, self.arm_base_offset)

    def _apply_action(self):
        """Override to align arm/gripper position target path with legacy ManipLoco."""
        self.torques = self._compute_torques(self.actions)
        self.robot.set_joint_effort_target(self.torques)

        pos_targets = self.dof_pos.clone()
        if hasattr(self, "gripper_pos_targets"):
            pos_targets[:, self.gripper_joint_ids] = self.gripper_pos_targets

        debug_enabled = bool(getattr(self, "teleop_debug", False)) and (int(getattr(self, "global_steps", 0)) % 20 == 0)
        arm_mode_used = "fallback"
        ik_delta_norm = None
        dpos_norm = None

        if self.teleop_mode and self.teleop_arm_control_mode == "joint":
            arm_mode_used = "joint"
            pos_targets[:, self.arm_joint_ids] = self.teleop_arm_joint_pos_targets
        elif (
            hasattr(self, "ee_pos")
            and hasattr(self, "curr_ee_goal_cart_world")
            and getattr(self, "ee_j_eef", None) is not None
        ):
            arm_mode_used = "ee_ik"
            dpos_norm = torch.norm(self.curr_ee_goal_cart_world - self.ee_pos, dim=-1)
            arm_pos_targets = self._get_arm_pos_targets()
            ik_delta_norm = torch.norm(arm_pos_targets - self.dof_pos[:, self.arm_joint_ids], dim=-1)
            pos_targets[:, self.arm_joint_ids] = arm_pos_targets
        else:
            # Fallback to old IsaacLab temporary path until full IK/Jacobian migration is complete.
            arm_mode_used = "fallback_target_pos"
            pos_targets[:, self.arm_joint_ids] = self.target_pos[:, self.arm_joint_ids]
            pos_targets[:, self.gripper_joint_ids] = self.target_pos[:, self.gripper_joint_ids]

        if debug_enabled:
            print("[IK DEBUG] ee_pos", self.ee_pos[0].detach().cpu().tolist())
            print("[IK DEBUG] curr_ee_goal_cart", self.curr_ee_goal_cart[0].detach().cpu().tolist())
            print("[IK DEBUG] curr_ee_goal_cart_world", self.curr_ee_goal_cart_world[0].detach().cpu().tolist())
            print("[IK DEBUG] dpos", (self.curr_ee_goal_cart_world - self.ee_pos)[0].detach().cpu().tolist())
            print("[IK DEBUG] dpos_norm", torch.norm(self.curr_ee_goal_cart_world - self.ee_pos, dim=-1)[0].item())
            print("[IK DEBUG] ee_orn", self.ee_orn[0].detach().cpu().tolist())
            print("[IK DEBUG] ee_goal_orn_quat", self.ee_goal_orn_quat[0].detach().cpu().tolist())
            print("[IK DEBUG] drot", orientation_error(
                self.ee_goal_orn_quat,
                self.ee_orn / torch.norm(self.ee_orn, dim=-1, keepdim=True).clamp_min(1e-8)
            )[0].detach().cpu().tolist())

            arm_now = self.dof_pos[0, self.arm_joint_ids].detach().cpu().tolist()
            arm_tgt = pos_targets[0, self.arm_joint_ids].detach().cpu().tolist()
            jac_shape = None if getattr(self, "ee_j_eef", None) is None else tuple(self.ee_j_eef.shape)
            fallback_reason = {
                "has_ee_pos": hasattr(self, "ee_pos"),
                "has_goal_world": hasattr(self, "curr_ee_goal_cart_world"),
                "has_jacobian": getattr(self, "ee_j_eef", None) is not None,
            }
            print(
                f"[teleop][debug] step={int(getattr(self, 'global_steps', 0))} "
                f"mode={arm_mode_used} teleop_mode={self.teleop_mode} teleop_arm_mode={self.teleop_arm_control_mode} "
                f"jacobian_shape={jac_shape}"
            )
            if arm_mode_used == "fallback_target_pos":
                print(f"[teleop][debug] fallback_reason={fallback_reason}")
            print(f"[teleop][debug] arm_now={arm_now}")
            print(f"[teleop][debug] arm_tgt={arm_tgt}")
            if dpos_norm is not None:
                print(f"[teleop][debug] dpos_norm={float(dpos_norm[0].detach().cpu()):.6f}")
            if ik_delta_norm is not None:
                print(f"[teleop][debug] ik_delta_norm={float(ik_delta_norm[0].detach().cpu()):.6f}")

        self.robot.set_joint_position_target(pos_targets)
        
        if int(getattr(self, "global_steps", 0)) % 5 == 0:
            self._draw_ee_goal_curr()        

    def _draw_points(self, points_w: torch.Tensor, colors, size: float = 12.0):
        """Draw world-frame points with IsaacSim debug draw."""
        if getattr(self, "_debug_draw", None) is None:
            return
        if points_w.numel() == 0:
            return

        pts = points_w.detach().cpu().float().tolist()
        cols = [colors for _ in pts]
        sizes = [size for _ in pts]
        self._debug_draw.draw_points(pts, cols, sizes)

    def _get_debug_draw(self):
        """Lazy acquire IsaacSim debug draw interface."""
        if hasattr(self, "_debug_draw"):
            return self._debug_draw

        self._debug_draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            print("[debug_draw] acquired from isaacsim.util.debug_draw")
        except Exception as e1:
            try:
                from omni.isaac.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                print("[debug_draw] acquired from omni.isaac.debug_draw")
            except Exception as e2:
                print(f"[debug_draw][warn] unavailable: {e1} / {e2}")
                self._debug_draw = False

        return self._debug_draw

    def _debug_draw_clear(self, clear_points=True, clear_lines=True):
        draw = self._get_debug_draw()
        if draw is None or draw is False:
            return

        try:
            if clear_points and hasattr(draw, "clear_points"):
                draw.clear_points()
        except Exception as e:
            print(f"[debug_draw][warn] clear_points failed: {e}")

        try:
            if clear_lines and hasattr(draw, "clear_lines"):
                draw.clear_lines()
        except Exception as e:
            print(f"[debug_draw][warn] clear_lines failed: {e}")

    def _draw_ee_goal_curr(self, env_ids=None):
        """Safe IsaacSim debug draw version.

        Yellow: curr_ee_goal_cart_world
        Blue:   ee_pos
        Cyan:   EE goal sphere center
        White:  robot root
        Green:  world origin

        Important:
        - no clear_points / clear_lines
        - no _refresh_ee_and_jacobian_for_ik inside draw
        - no axes/lines first
        - draw only low frequency from caller
        """
        draw = self._get_debug_draw()
        if draw is None or draw is False:
            return

        step = int(getattr(self, "global_steps", 0))

        # clear less frequently to avoid viewer/debug-draw stall
        if step % 5 == 0:
            self._debug_draw_clear(clear_points=True, clear_lines=True)
            
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        points = []
        colors = []
        sizes = []

        def add_points(tensor_points, color, size):
            if tensor_points is None:
                return
            if tensor_points.numel() == 0:
                return
            pts = tensor_points.detach().cpu().float().tolist()
            points.extend(pts)
            colors.extend([color] * len(pts))
            sizes.extend([float(size)] * len(pts))

        # Yellow: target EE goal in world.
        if hasattr(self, "curr_ee_goal_cart_world"):
            add_points(self.curr_ee_goal_cart_world[env_ids], (1.0, 1.0, 0.0, 1.0), 12.0)

        # Blue: measured EE position. Do NOT refresh here; draw should be read-only.
        if hasattr(self, "ee_pos"):
            add_points(self.ee_pos[env_ids], (0.0, 0.0, 1.0, 1.0), 10.0)

        # Cyan: old get_ee_goal_spherical_center().
        if hasattr(self, "get_ee_goal_spherical_center"):
            try:
                add_points(self.get_ee_goal_spherical_center()[env_ids], (0.0, 1.0, 1.0, 1.0), 10.0)
            except Exception as e:
                if int(getattr(self, "global_steps", 0)) % 200 == 0:
                    print(f"[debug_draw][warn] goal center draw skipped: {e}")

        # White: robot root.
        try:
            if hasattr(self.robot.data, "root_pos_w"):
                add_points(self.robot.data.root_pos_w[env_ids], (1.0, 1.0, 1.0, 1.0), 10.0)
        except Exception:
            pass

        # Green: world origin.
        add_points(torch.zeros(1, 3, device=self.device), (0.0, 1.0, 0.0, 1.0), 14.0)

        if len(points) == 0:
            return

        try:
            draw.draw_points(points, colors, sizes)
        except Exception as e:
            print(f"[debug_draw][error] draw_points failed: {e}")
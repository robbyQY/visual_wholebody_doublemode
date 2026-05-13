from __future__ import annotations
import math
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils

from legged_gym.utils.isaaclab_math import quat_rotate_inverse, quat_apply, quat_from_euler_xyz, euler_from_quat, torch_rand_float, wrap_to_pi
from .legged_robot_isaaclab_config import LeggedRobotIsaacLabCfg


class LeggedRobotIsaacLab(DirectRLEnv):
    """Core IsaacLab port of the uploaded legged_gym BaseTask/LeggedRobot simulator layer.

    This replaces gymapi/gymtorch calls with IsaacLab DirectRLEnv callbacks.
    Reward/observation details are intentionally kept simple here so the simulator
    bridge can be verified first.
    """
    cfg: LeggedRobotIsaacLabCfg

    def __init__(self, cfg: LeggedRobotIsaacLabCfg, render_mode: str | None = None, **kwargs):
        if cfg.robot_urdf_path:
            cfg.robot.spawn.robot_urdf_path = cfg.robot_urdf_path
        if not cfg.robot.spawn.robot_urdf_path:
            raise ValueError(
                "cfg.robot_usd_path is empty. Convert your URDF to USD and set LEGGED_GYM_ROBOT_USD=/abs/path/robot.usd "
                "or pass cfg.robot_usd_path before constructing the env."
            )
        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = int(cfg.action_space)
        self.num_obs = int(cfg.observation_space)
        print("self.cfg.decimationnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn:", self.cfg.decimation)
        print("self.cfg.sim.dtself.cfg.sim.dtself.cfg.sim.dtself.cfg.sim.dtself.cfg.sim.dtself.cfg.sim.dt: ", self.cfg.sim.dt)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # self.dt = self.cfg.sim.dt * 4
        print("step_dt: ", self.step_dt)
        # self.max_episode_length_s = float(cfg.episode_length_s)
        # self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        self.legacy_max_episode_length_s = float(cfg.episode_length_s)
        self.legacy_max_episode_length = int(math.ceil(self.legacy_max_episode_length_s / self.step_dt))        

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras = {}

        ######################################################################
        scale = self.cfg.control.action_scale
        if isinstance(scale, (float, int)):
            self.action_scale_tensor = torch.full((self.num_actions,), float(scale), device=self.device)
        else:
            self.action_scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)

        assert self.action_scale_tensor.numel() == self.num_actions
        #########################################################################

        self._build_joint_maps()
        self._init_policy_compat_buffers()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.sim.set_camera_view((4.0, 0.0, 2.0), (0.0, 0.0, 0.5))

    def _build_joint_maps(self):
        self.dof_names = list(self.robot.data.joint_names)
        self.body_names = list(self.robot.data.body_names)
        self.dof_names_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self.body_names_to_idx = {name: i for i, name in enumerate(self.body_names)}
        self.num_dofs = len(self.dof_names)
        self.num_bodies = len(self.body_names)

        self.policy_joint_ids = torch.tensor(
            [self.dof_names_to_idx[name] for name in self.cfg.policy_joint_names if name in self.dof_names_to_idx],
            dtype=torch.long, device=self.device)
        if self.policy_joint_ids.numel() != len(self.cfg.policy_joint_names):
            missing = [n for n in self.cfg.policy_joint_names if n not in self.dof_names_to_idx]
            raise RuntimeError(f"Missing policy joints in USD articulation: {missing}. Available={self.dof_names}")

        self.feet_indices = torch.tensor(
            [self.body_names_to_idx[name] for name in self.cfg.foot_body_names if name in self.body_names_to_idx],
            dtype=torch.long, device=self.device)
        self.termination_contact_indices = torch.tensor(
            [self.body_names_to_idx[name] for name in self.cfg.terminate_body_names if name in self.body_names_to_idx],
            dtype=torch.long, device=self.device)

        self.default_dof_pos = self.robot.data.default_joint_pos[0].clone()
        for name, value in self.cfg.default_joint_angles.items():
            if name in self.dof_names_to_idx:
                self.default_dof_pos[self.dof_names_to_idx[name]] = float(value)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        self.p_gains = torch.zeros(self.num_dofs, device=self.device)
        self.d_gains = torch.zeros(self.num_dofs, device=self.device)
        for i, name in enumerate(self.dof_names):
            for key, val in self.cfg.control.stiffness.items():
                if key == name or key in name:
                    self.p_gains[i] = float(val)
            for key, val in self.cfg.control.damping.items():
                if key == name or key in name:
                    self.d_gains[i] = float(val)
        ################################################################################
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.arm_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.arm_joint_names if n in self.dof_names_to_idx],
            dtype=torch.long,
            device=self.device,
        )

        self.gripper_joint_names = ["jointGripper"]
        self.gripper_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.gripper_joint_names if n in self.dof_names_to_idx],
            dtype=torch.long,
            device=self.device,
        )        
        print("[arm_joint_ids]", self.arm_joint_ids.detach().cpu().tolist())
        print("[gripper_joint_ids]", self.gripper_joint_ids.detach().cpu().tolist())        
        ############################################        
        self.policy_all_joint_names = list(self.cfg.policy_joint_names)
        if "jointGripper" in self.dof_names_to_idx and "jointGripper" not in self.policy_all_joint_names:
            self.policy_all_joint_names.append("jointGripper")

        self.policy_all_joint_ids = torch.tensor(
            [self.dof_names_to_idx[n] for n in self.policy_all_joint_names],
            dtype=torch.long,
            device=self.device,
        )#  

    def _env_to_policy_all(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert a sim-order 19-DOF tensor to policy-all order.

        Only use this for tensors whose last dim is num_dofs=19:
        - dof_pos
        - dof_vel
        - default_dof_pos
        - torques if needed

        Do NOT use this for action tensors, because actions are 18-dim.
        """
        if vec.shape[-1] != self.num_dofs:
            raise RuntimeError(
                f"_env_to_policy_all expects last dim == num_dofs ({self.num_dofs}), "
                f"but got shape {tuple(vec.shape)}. "
                "Do not call _env_to_policy_all() on 18-dim action tensors."
            )
        return vec[:, self.policy_all_joint_ids]

    def _env_to_policy_dog(self, vec: torch.Tensor) -> torch.Tensor:
        return vec[:, self.policy_joint_ids[:12]]

    def _policy_to_env_all(self, actions: torch.Tensor) -> torch.Tensor:
        # Input is 18 policy actions. Return same 18 policy order here.
        # Actual sim-order mapping happens via self.policy_joint_ids in _compute_torques.
        return actions               

    def _init_policy_compat_buffers(self):
        # These fields mimic old ManipLoco policy-side buffers.
        self.num_proprio = int(getattr(self.cfg, "num_proprio", 72))
        self.num_priv = int(getattr(self.cfg, "num_priv", 18))
        self.history_len = int(getattr(self.cfg, "history_len", 10))
        self.num_gripper_joints = int(getattr(self.cfg, "num_gripper_joints", 1))

        self.obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        self.privileged_obs_buf = None

        self.obs_history_buf = torch.zeros(
            self.num_envs, self.history_len, self.num_proprio, device=self.device
        )

        # old action_history_buf stores env-order full policy actions.
        # action_delay=3 is in metadata, but old auto mode uses -1 early in training/play.
        self.action_delay = int(getattr(self.cfg, "action_delay", 3))
        self.action_delay_mode = getattr(self.cfg, "action_delay_mode", "auto")
        self.action_history_buf = torch.zeros(
            self.num_envs, max(self.action_delay + 1, 4), self.num_actions, device=self.device
        )

        self.global_steps = 0
        self.last_torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

        # Old obs scales used by ManipLoco.compute_observations().
        self.obs_scales = type("ObsScales", (), {})()
        self.obs_scales.ang_vel = 0.25
        self.obs_scales.dof_pos = 1.0
        self.obs_scales.dof_vel = 0.05

        # Old command scaling for [lin_x, lin_y, yaw].
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)

        # Domain-rand priv obs. In play, use neutral values.
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self.friction_coeffs_tensor = torch.ones(self.num_envs, 1, device=self.device)
        self.motor_strength = torch.ones(self.num_envs, 12, device=self.device)
        self.mass_params_tensor = torch.tensor(
            [[
                1.0417996644973755,
                0.027897033840417862,
                -0.004937552381306887,
                0.0034558435436338186,
                0.004164694342762232,
            ]],
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.friction_coeffs_tensor = torch.tensor(
            [[0.520315408706665]],
            device=self.device,
        ).repeat(self.num_envs, 1)

        motor_strength_minus_1 = torch.tensor(
            [[
                -0.050519704818725586,
                -0.002183079719543457,
                0.015573859214782715,
                0.08771336078643799,
                0.04324972629547119,
                0.051445960998535156,
                -0.036211252212524414,
                -0.06478011608123779,
                0.02942824363708496,
                0.09396469593048096,
                0.07743573188781738,
                -0.003672182559967041,
            ]],
            device=self.device,
        )

        self.motor_strength = 1.0 + motor_strength_minus_1        

        # EE goal obs mode "command": checkpoint expects curr_ee_goal_cart.
        # self.curr_ee_goal_cart = torch.tensor(
        #     [0.2, 0.0, 0.2], device=self.device
        # ).repeat(self.num_envs, 1)
        self.curr_ee_goal_cart = torch.tensor(
            [0.4619397819042206, 0.0, 0.19134172797203064],
            device=self.device,
        ).repeat(self.num_envs, 1)        
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        # mixed_height_reference bit.
        self.goal_height_follow_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # gait obs: gait_indices + 4 clock inputs.
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)

        # foot contacts. Until force sensors are ported, use contact sensor if available.
        self.foot_contacts_from_sensor = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        # Noise disabled for play.
        self.add_noise = False

    @property
    def root_states(self):
        # legacy-style [pos(3), quat wxyz(4), lin_vel_w(3), ang_vel_w(3)]
        return torch.cat((
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
        ), dim=-1)

    @property
    def dof_pos(self):
        return self.robot.data.joint_pos

    @property
    def dof_vel(self):
        return self.robot.data.joint_vel

    @property
    def base_quat(self):
        return self.robot.data.root_quat_w

    @property
    def base_lin_vel(self):
        return self.robot.data.root_lin_vel_b

    @property
    def base_ang_vel(self):
        return self.robot.data.root_ang_vel_b

    @property
    def projected_gravity(self):
        return self.robot.data.projected_gravity_b

    @property
    def contact_forces(self):
        return self.contact_sensor.data.net_forces_w   

    def _debug_enabled(self) -> bool:
        """Print full obs/action debug only for the first few env steps."""
        return int(getattr(self, "global_steps", 0)) < 20

    def _debug_print_tensor(self, name: str, x: torch.Tensor | None, max_full_elems: int = 300):
        """Print tensor shape, stats, and full env0 value when small enough."""
        if x is None:
            print(f"[NEW DEBUG] {name}: None")
            return

        xd = x.detach()
        shape = tuple(xd.shape)

        if xd.numel() == 0:
            print(f"[NEW DEBUG] {name}: shape={shape}, numel=0")
            return

        xf = xd.float().reshape(-1)
        x_min = float(xf.min().cpu())
        x_max = float(xf.max().cpu())
        x_mean = float(xf.mean().cpu())

        print(
            f"[NEW DEBUG] {name}: "
            f"shape={shape}, numel={xd.numel()}, "
            f"min={x_min:+.6f}, max={x_max:+.6f}, mean={x_mean:+.6f}"
        )

        if xd.ndim >= 2:
            env0 = xd[0].reshape(-1)
        else:
            env0 = xd.reshape(-1)

        if env0.numel() <= max_full_elems:
            print(f"[NEW DEBUG] {name}[0] = {env0.cpu().tolist()}")
        else:
            print(
                f"[NEW DEBUG] {name}[0][:40] = {env0[:40].cpu().tolist()} "
                f"... total_dim={env0.numel()}"
            )

    def _pre_physics_step(self, actions: torch.Tensor):
        debug_this_step = self._debug_enabled()

        if debug_this_step:
            print("\n" + "#" * 120)
            print(f"[NEW ACTION DEBUG BEGIN] global_steps={int(self.global_steps)}")
            print("#" * 120)
            self._debug_print_tensor("actions_input_from_policy_raw_policy_order", actions)

        if actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {actions.shape[-1]}")

        clip = float(self.cfg.control.clip_actions)

        # Match old ManipLoco step(): actions[:, 12:] = 0.0
        actions_before_arm_zero = actions.clone()
        actions = actions.clone()
        actions[:, 12:] = 0.0

        actions_before_clip = actions.clone()
        actions = torch.clip(actions, -clip, clip).to(self.device)

        if debug_this_step:
            print(f"[NEW DEBUG] cfg.control.clip_actions = {clip}")
            self._debug_print_tensor("actions_before_arm_zero_policy_order", actions_before_arm_zero)
            self._debug_print_tensor("actions_after_arm_zero_policy_order", actions_before_clip)
            self._debug_print_tensor("actions_policy_order_after_clip", actions)
            self._debug_print_tensor("actions_leg12_after_clip_policy_order", actions[:, :12])
            self._debug_print_tensor("actions_arm6_after_clip_policy_order", actions[:, 12:])

        # Old ManipLoco action delay buffer.
        action_history_before = self.action_history_buf.clone() if debug_this_step else None

        self.action_history_buf = torch.cat(
            [
                self.action_history_buf[:, 1:],
                actions[:, None, :],
            ],
            dim=1,
        )

        if debug_this_step:
            self._debug_print_tensor("action_history_buf_before_update", action_history_before)
            self._debug_print_tensor("action_history_buf_after_update", self.action_history_buf)
            self._debug_print_tensor("action_history_latest", self.action_history_buf[:, -1])
            if self.action_history_buf.shape[1] >= 2:
                self._debug_print_tensor("action_history_previous", self.action_history_buf[:, -2])

        mode = getattr(self, "action_delay_mode", "auto")

        if mode == "undelayed":
            effective_actions = self.action_history_buf[:, -1]
        elif mode == "delayed":
            effective_actions = self.action_history_buf[:, -2]
        else:
            # Match old play behavior: early phase uses undelayed.
            if self.global_steps < 10000 * 24:
                effective_actions = self.action_history_buf[:, -1]
            else:
                effective_actions = self.action_history_buf[:, -2]

        self.actions = effective_actions.clone()

        # if debug_this_step:
        #     print(f"[NEW DEBUG] action_delay_mode = {mode}")
        #     self._debug_print_tensor("actions_final_after_delay_used_for_torque", self.actions)

        # # self.torques = self._compute_torques(self.actions)

        # if debug_this_step:
        #     self._debug_print_tensor("torques_after_compute_torques_sim_order_19", self.torques)
        #     if hasattr(self, "target_pos"):
        #         self._debug_print_tensor("target_pos_after_compute_torques_sim_order_19", self.target_pos)
        #     print("#" * 120)
        #     print(f"[NEW ACTION DEBUG END] global_steps={int(self.global_steps)}")
        #     print("#" * 120 + "\n")

    def _apply_action(self):
        # Full sim-order joint position target.
        # Shape: [num_envs, num_dofs].
        # It is generated in _compute_torques(actions).
        debug_this_step = self._debug_enabled()

        self.torques = self._compute_torques(self.actions)
        # Optional effort target. For old ManipLoco-like behavior,
        # arm/gripper torque is zeroed in _compute_torques().
        self.robot.set_joint_effort_target(self.torques)
                
        if debug_this_step:
            print("\n[NEW APPLY ACTION DEBUG]")
            self._debug_print_tensor("target_pos_sent_to_isaaclab_sim_order_19", self.target_pos)
            self._debug_print_tensor("torques_sent_to_isaaclab_sim_order_19", self.torques)
                    
        # arm/gripper: only position target
        pos_targets = self.dof_pos.clone()
        pos_targets[:, self.arm_joint_ids] = self.target_pos[:, self.arm_joint_ids]
        pos_targets[:, self.gripper_joint_ids] = self.target_pos[:, self.gripper_joint_ids]

        self.robot.set_joint_position_target(pos_targets)

        self.global_steps += 1    

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute sim-order effort targets from policy-order actions.

        actions:
            [num_envs, 18], policy joint order:
            12 leg joints + 6 arm joints.

        target_pos:
            [num_envs, 19], IsaacLab sim joint order:
            12 legs + 6 arm + jointGripper.
        """
        debug_this_step = self._debug_enabled()

        if actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {actions.shape[-1]}")

        # Per-action scale from checkpoint metadata.
        # Shape: [num_envs, 18]
        scaled = actions * self.action_scale_tensor.unsqueeze(0)

        # Start from default pose in sim joint order.
        # Shape: [num_envs, num_dofs]
        target_pos = self.default_dof_pos.clone()

        # Write policy actions into corresponding sim joint ids.
        joint_ids = self.policy_joint_ids
        target_pos[:, joint_ids] = (
            self.default_dof_pos[:, joint_ids]
            + scaled[:, : joint_ids.numel()]
        )

        # Keep for IsaacLab position drive.
        self.target_pos = target_pos

        if self.cfg.control.control_type == "P":
            torques_unclipped = (
                self.p_gains.unsqueeze(0) * (target_pos - self.dof_pos)
                - self.d_gains.unsqueeze(0) * self.dof_vel
            )
        elif self.cfg.control.control_type == "T":
            torques_unclipped = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
            torques_unclipped[:, joint_ids] = scaled[:, : joint_ids.numel()]
        else:
            raise NotImplementedError(f"Unsupported control type: {self.cfg.control.control_type}")

        torques_after_arm_zero = torques_unclipped.clone()

        # Match old ManipLoco more closely:
        # old ManipLoco zeroed the arm torque and used position target for arm/gripper.
        if hasattr(self, "arm_joint_ids") and self.arm_joint_ids.numel() > 0:
            torques_after_arm_zero[:, self.arm_joint_ids] = 0.0
        if hasattr(self, "gripper_joint_ids") and self.gripper_joint_ids.numel() > 0:
            torques_after_arm_zero[:, self.gripper_joint_ids] = 0.0

        # Constant clamp for bring-up.
        torques = torch.clamp(torques_after_arm_zero, -600.0, 600.0)

        if debug_this_step:
            print("\n[NEW TORQUE DEBUG]")
            print(f"[NEW DEBUG] control_type = {self.cfg.control.control_type}")
            self._debug_print_tensor("actions_input_to_compute_torques_policy_order_18", actions)
            self._debug_print_tensor("action_scale_tensor_18", self.action_scale_tensor)
            self._debug_print_tensor("scaled_actions_policy_order_18", scaled)

            print("[NEW DEBUG] joint_ids policy->sim =", joint_ids.detach().cpu().tolist())
            print("[NEW DEBUG] policy_joint_names =", self.cfg.policy_joint_names)

            self._debug_print_tensor("default_dof_pos_sim_order_19", self.default_dof_pos)
            self._debug_print_tensor("dof_pos_current_sim_order_19", self.dof_pos)
            self._debug_print_tensor("dof_vel_current_sim_order_19", self.dof_vel)
            self._debug_print_tensor("target_pos_sim_order_19", target_pos)
            self._debug_print_tensor("target_offset_sim_order_19", target_pos - self.default_dof_pos)

            self._debug_print_tensor("p_gains_sim_order_19", self.p_gains)
            self._debug_print_tensor("d_gains_sim_order_19", self.d_gains)
            self._debug_print_tensor("torques_unclipped_sim_order_19", torques_unclipped)
            self._debug_print_tensor("torques_after_arm_gripper_zero_sim_order_19", torques_after_arm_zero)
            self._debug_print_tensor("torques_final_clipped_sim_order_19", torques)

        return torques

    # def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
    #     scaled = actions * self.action_scale_tensor.unsqueeze(0)

    #     target_pos = self.default_dof_pos.clone()
    #     target_pos[:, self.policy_joint_ids] = (
    #         self.default_dof_pos[:, self.policy_joint_ids]
    #         + scaled[:, : self.policy_joint_ids.numel()]
    #     )

    #     self.target_pos = target_pos

    #     if not hasattr(self, "_target_debug_counter"):
    #         self._target_debug_counter = 0
    #     self._target_debug_counter += 1

    #     if self._target_debug_counter % 50 == 0:
    #         offsets = (self.target_pos - self.default_dof_pos)[0, self.policy_joint_ids[:12]]
    #         print("[target offset rad leg]", offsets.detach().cpu().tolist())

    #     return torch.zeros(self.num_envs, self.num_dofs, device=self.device)    

    def _get_body_orientation(self, return_yaw: bool = False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)
        return body_angles if return_yaw else body_angles[:, :-1]

    def _update_policy_aux_obs(self):
        # # Foot contact approximation from ContactSensor.
        # # If contact force body order is uncertain, use robot body feet_indices as first approximation.
        # try:
        #     forces = self.contact_forces[:, self.feet_indices, :]
        #     self.foot_contacts_from_sensor = torch.norm(forces, dim=-1) > 1.5
        # except Exception:
        #     self.foot_contacts_from_sensor = torch.zeros(
        #         self.num_envs, 4, dtype=torch.bool, device=self.device
        #     )

        # # Gait clock. For first policy test, fixed frequency 2.0 Hz is enough.
        # # dt is env step dt, not physics dt.
        # freq = 2.0
        # self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * freq, 1.0)
        
        # 先强制 foot contact 和 old debug 一致
        # self.foot_contacts_from_sensor[:] = torch.tensor(
        #     [False, False, True, False],
        #     device=self.device,
        #     dtype=torch.bool,
        # ).unsqueeze(0)
        net_forces = self.contact_sensor.data.net_forces_w

        if not hasattr(self, "_printed_contact_shape"):
            self._printed_contact_shape = True
            print("[contact net_forces_w shape]", tuple(net_forces.shape))
            if hasattr(self.contact_sensor, "body_names"):
                print("[contact sensor body_names]", self.contact_sensor.body_names)

        # 如果 ContactSensor 只监控 4 个 foot body，直接用它本身
        if net_forces.shape[1] == 4:
            forces = net_forces

        # 如果 ContactSensor 监控了所有 robot body，才可以用 robot feet_indices
        elif net_forces.shape[1] == len(self.body_names):
            forces = net_forces[:, self.feet_indices, :]

        else:
            print("[WARN] unexpected contact force shape:", tuple(net_forces.shape))
            print("[WARN] robot feet_indices:", self.feet_indices.detach().cpu().tolist())
            forces = torch.zeros(self.num_envs, 4, 3, device=self.device)

        force_norm = torch.norm(forces, dim=-1)
        self.foot_contacts_from_sensor = force_norm > 1.5

        if int(self.global_steps) < 20:
            print("[feet contact forces]", forces[0].detach().cpu().tolist())
            print("[feet contact norm]", force_norm[0].detach().cpu().tolist())
            print("[foot contacts]", self.foot_contacts_from_sensor[0].detach().cpu().tolist())              

        # 先固定 gait / clock，不要推进
        self.gait_indices[:] = 0.0
        self.clock_inputs[:] = torch.tensor(
            [-8.742277657347586e-08, 0.0, 0.0, -8.742277657347586e-08],
            device=self.device,
        ).unsqueeze(0)
        
        # phase = 2.0 * torch.pi * self.gait_indices
        # self.clock_inputs = torch.stack(
        #     [
        #         torch.sin(phase),
        #         torch.cos(phase),
        #         torch.sin(phase + torch.pi),
        #         torch.cos(phase + torch.pi),
        #     ],
        #     dim=-1,
        # )

    def _get_observations(self):
        self._update_policy_aux_obs()

        ee_goal_local_cart = self.curr_ee_goal_cart

        obs_body_orientation = self._get_body_orientation()
        # dim 2

        obs_base_ang_vel = self.base_ang_vel * self.obs_scales.ang_vel
        # dim 3

        obs_dof_pos = self._env_to_policy_all(
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        )[:, :-self.num_gripper_joints]
        # dim 18

        obs_dof_vel = self._env_to_policy_all(
            self.dof_vel * self.obs_scales.dof_vel
        )[:, :-self.num_gripper_joints]
        # dim 18

        obs_last_leg_actions = self.action_history_buf[:, -1, :12]
        # dim 12

        obs_foot_contacts = self.foot_contacts_from_sensor
        # dim 4

        obs_commands = self.commands[:, :3] * self.commands_scale
        # dim 3

        obs_ee_goal_local_cart = ee_goal_local_cart
        # dim 3

        obs_ee_goal_orientation_dummy = 0.0 * self.curr_ee_goal_sphere
        # dim 3

        obs_terms_named = [
            ("body_orientation_2", obs_body_orientation),
            ("base_ang_vel_scaled_3", obs_base_ang_vel),
            ("dof_pos_minus_default_scaled_policy_order_18", obs_dof_pos),
            ("dof_vel_scaled_policy_order_18", obs_dof_vel),
            ("last_leg_actions_policy_order_12", obs_last_leg_actions),
            ("foot_contacts_4", obs_foot_contacts),
            ("commands_scaled_3", obs_commands),
            ("ee_goal_local_cart_3", obs_ee_goal_local_cart),
            ("ee_goal_orientation_dummy_3", obs_ee_goal_orientation_dummy),
        ]

        if getattr(self.cfg, "mixed_height_reference", True):
            obs_goal_height_mask = self.goal_height_follow_mask.float().unsqueeze(1)
            obs_terms_named.append(("goal_height_follow_mask_1", obs_goal_height_mask))

        obs_buf = torch.cat([x for _, x in obs_terms_named], dim=-1)

        if getattr(self.cfg, "observe_gait_commands", True):
            obs_gait_indices = self.gait_indices.unsqueeze(1)
            obs_clock_inputs = self.clock_inputs

            obs_terms_named.append(("gait_indices_1", obs_gait_indices))
            obs_terms_named.append(("clock_inputs_4", obs_clock_inputs))

            obs_buf = torch.cat(
                [
                    obs_buf,
                    obs_gait_indices,
                    obs_clock_inputs,
                ],
                dim=-1,
            )

        # -----------------------------
        # Full obs debug, first 5 steps
        # -----------------------------
        if self._debug_enabled():
            print("\n" + "=" * 120)
            print(f"[NEW OBS DEBUG] global_steps={int(self.global_steps)}")
            print("=" * 120)

            obs_start = 0
            for name, term in obs_terms_named:
                dim = term.shape[-1]
                obs_end = obs_start + dim

                print(f"\n[NEW OBS TERM] {name}: obs_slice=[{obs_start}:{obs_end}], dim={dim}")
                self._debug_print_tensor(name, term)

                obs_start = obs_end

            print(f"\n[NEW OBS CAT] proprio obs_buf dim = {obs_buf.shape[-1]}")
            self._debug_print_tensor("obs_buf_proprio_72", obs_buf)

            self._debug_print_tensor("raw_root_states", self.root_states)
            self._debug_print_tensor("root_pos_w", self.robot.data.root_pos_w)
            self._debug_print_tensor("root_quat_w", self.robot.data.root_quat_w)
            self._debug_print_tensor("root_lin_vel_w", self.robot.data.root_lin_vel_w)
            self._debug_print_tensor("root_ang_vel_w", self.robot.data.root_ang_vel_w)
            self._debug_print_tensor("base_lin_vel_b", self.base_lin_vel)
            self._debug_print_tensor("base_ang_vel_b", self.base_ang_vel)
            self._debug_print_tensor("projected_gravity_b", self.projected_gravity)
            self._debug_print_tensor("dof_pos_sim_order_19", self.dof_pos)
            self._debug_print_tensor("dof_vel_sim_order_19", self.dof_vel)
            self._debug_print_tensor("default_dof_pos_sim_order_19", self.default_dof_pos)

            print("[NEW DEBUG] dof_names sim order =", self.dof_names)
            print("[NEW DEBUG] policy_joint_names =", self.cfg.policy_joint_names)
            print("[NEW DEBUG] policy_joint_ids =", self.policy_joint_ids.detach().cpu().tolist())
            print("[NEW DEBUG] policy_all_joint_names =", self.policy_all_joint_names)
            print("[NEW DEBUG] policy_all_joint_ids =", self.policy_all_joint_ids.detach().cpu().tolist())

        # Sanity: checkpoint expects proprio dim 72.
        if obs_buf.shape[-1] != self.num_proprio:
            raise RuntimeError(
                f"Expected proprio dim {self.num_proprio}, got {obs_buf.shape[-1]}. "
                f"Check obs layout."
            )

        priv_buf = torch.cat(
            [
                self.mass_params_tensor,          # 4
                self.friction_coeffs_tensor,      # 1
                self.motor_strength[:, :12] - 1,  # 12
                # torch.zeros(self.num_envs, 1, device=self.device),  # pad to 18
            ],
            dim=-1,
        )

        if priv_buf.shape[-1] != self.num_priv:
            raise RuntimeError(f"Expected priv dim {self.num_priv}, got {priv_buf.shape[-1]}")

        if self._debug_enabled():
            print("\n[NEW PRIV DEBUG]")
            self._debug_print_tensor("mass_params_tensor_4", self.mass_params_tensor)
            self._debug_print_tensor("friction_coeffs_tensor_1", self.friction_coeffs_tensor)
            self._debug_print_tensor("motor_strength_minus_1_leg_12", self.motor_strength[:, :12] - 1)
            self._debug_print_tensor("priv_pad_1", torch.zeros(self.num_envs, 1, device=self.device))
            self._debug_print_tensor("priv_buf_18", priv_buf)
            self._debug_print_tensor("obs_history_buf_before_update", self.obs_history_buf)
            self._debug_print_tensor(
                "obs_history_flat_before_update",
                self.obs_history_buf.reshape(self.num_envs, -1),
            )

        self.obs_buf = torch.cat(
            [
                obs_buf,
                priv_buf,
                self.obs_history_buf.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

        if self.obs_buf.shape[-1] != self.num_obs:
            raise RuntimeError(
                f"Expected obs dim {self.num_obs}, got {self.obs_buf.shape[-1]}"
            )

        if self._debug_enabled():
            print("\n[NEW FINAL OBS BEFORE HISTORY UPDATE]")
            self._debug_print_tensor("self.obs_buf_before_clip_input_to_policy", self.obs_buf)
            print(f"[NEW FINAL OBS DIM] self.obs_buf.shape = {tuple(self.obs_buf.shape)}")

        # Update history after constructing obs, matching old code ordering.
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.history_len, dim=1),
            torch.cat(
                [
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1),
                ],
                dim=1,
            ),
        )

        self.obs_buf = torch.clip(
            self.obs_buf,
            -self.cfg.control.clip_observations,
            self.cfg.control.clip_observations,
        )

        if self._debug_enabled():
            print("\n[NEW FINAL OBS AFTER CLIP]")
            print(f"[NEW DEBUG] clip_observations = {self.cfg.control.clip_observations}")
            self._debug_print_tensor("self.obs_buf_after_clip_return_policy", self.obs_buf)
            print("=" * 120 + "\n")

        return {"policy": self.obs_buf}

    def _get_rewards(self):
        lin_err = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=-1)
        yaw_err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        action_penalty = 0.01 * torch.sum(self.actions ** 2, dim=-1)
        self.rew_buf = torch.exp(-lin_err / 0.25) + 0.5 * torch.exp(-yaw_err / 0.25) - action_penalty
        return self.rew_buf

    def _get_dones(self):
        if self.termination_contact_indices.numel() > 0:
            bad_contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1) > 1.0, dim=1)
        else:
            bad_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf >= self.legacy_max_episode_length
        return bad_contact, self.time_out_buf

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        joint_pos = self.default_dof_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), self.device)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return
        c = self.cfg.commands
        self.commands[env_ids, 0] = torch_rand_float(c.lin_vel_x[0], c.lin_vel_x[1], (len(env_ids), 1), self.device).squeeze(-1)
        if self.commands.shape[1] > 1:
            self.commands[env_ids, 1] = torch_rand_float(c.lin_vel_y[0], c.lin_vel_y[1], (len(env_ids), 1), self.device).squeeze(-1)
        if self.commands.shape[1] > 2:
            self.commands[env_ids, 2] = torch_rand_float(c.ang_vel_yaw[0], c.ang_vel_yaw[1], (len(env_ids), 1), self.device).squeeze(-1)

    # legacy API conveniences used by old scripts
    def get_observations(self):
        # return self._get_observations()["policy"]
        return self._get_observations()

    def get_privileged_observations(self):
        return None

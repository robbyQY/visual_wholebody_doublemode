from __future__ import annotations
import math
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils

from legged_gym.utils.isaaclab_math import quat_rotate_inverse, quat_apply, quat_from_euler_xyz, euler_from_quat, torch_rand_float, wrap_to_pi
from .legged_robot_isaaclab_config import LeggedRobotIsaacLabCfg
import numpy as np
from rsl_rl.utils import resolve_schedule_value

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
        print("self.cfg.decimation:", self.cfg.decimation)
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

        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.gait_frequencies = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)        


        schedule_counter = float(getattr(self.cfg.commands, "curriculum_playback_counter", 0.0) or 0.0)
        schedule_total_iterations = getattr(self.cfg.commands, "curriculum_playback_total_iterations", None)
        lin_vel_x_min = resolve_schedule_value(
            self.cfg.commands.lin_vel_x_min_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        lin_vel_x_max = resolve_schedule_value(
            self.cfg.commands.lin_vel_x_max_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        ang_vel_yaw_max = resolve_schedule_value(
            self.cfg.commands.ang_vel_yaw_schedule,
            counter=schedule_counter,
            default_end_iter=schedule_total_iterations,
        )
        self.command_ranges = {
            "lin_vel_x": [lin_vel_x_min, lin_vel_x_max],
            "ang_vel_yaw": [-ang_vel_yaw_max, ang_vel_yaw_max],
        }

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
        self.obs_scales.ang_vel = 1.0
        self.obs_scales.dof_pos = 1.0
        self.obs_scales.dof_vel = 0.05

        # Old command scaling for [lin_x, lin_y, yaw].
        # self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.commands_scale = torch.tensor([1.0, 1.0, 1.0], device=self.device)

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

    def _pre_physics_step(self, actions: torch.Tensor):

        if actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {actions.shape[-1]}")

        clip = float(self.cfg.control.clip_actions)

        # Match old ManipLoco step(): actions[:, 12:] = 0.0
        actions_before_arm_zero = actions.clone()
        actions = actions.clone()
        actions[:, 12:] = 0.0

        actions_before_clip = actions.clone()
        actions = torch.clip(actions, -clip, clip).to(self.device)

        self.action_history_buf = torch.cat(
            [
                self.action_history_buf[:, 1:],
                actions[:, None, :],
            ],
            dim=1,
        )

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
        self.global_steps += 1

    def _apply_action(self):
        # Full sim-order joint position target.
        # Shape: [num_envs, num_dofs].
        # It is generated in _compute_torques(actions).

        self.torques = self._compute_torques(self.actions)
        # Optional effort target. For old ManipLoco-like behavior,
        # arm/gripper torque is zeroed in _compute_torques().
        self.robot.set_joint_effort_target(self.torques)
                    
        # arm/gripper: only position target
        pos_targets = self.dof_pos.clone()
        pos_targets[:, self.arm_joint_ids] = self.target_pos[:, self.arm_joint_ids]
        pos_targets[:, self.gripper_joint_ids] = self.target_pos[:, self.gripper_joint_ids]

        self.robot.set_joint_position_target(pos_targets)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute sim-order effort targets from policy-order actions.

        actions:
            [num_envs, 18], policy joint order:
            12 leg joints + 6 arm joints.

        target_pos:
            [num_envs, 19], IsaacLab sim joint order:
            12 legs + 6 arm + jointGripper.
        """

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

        return torques 

    def _get_body_orientation(self, return_yaw: bool = False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)
        return body_angles if return_yaw else body_angles[:, :-1]

    def _update_policy_aux_obs(self):
        # # Foot contact approximation from ContactSensor.
        # # If contact force body order is uncertain, use robot body feet_indices as first approximation.

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

        self._step_contact_targets()
        
    def _get_observations(self):
        self._update_policy_aux_obs()

        ee_goal_obs_mode = getattr(self.cfg, "ee_goal_obs_mode", None)
        if ee_goal_obs_mode is None and hasattr(self.cfg, "env"):
            ee_goal_obs_mode = getattr(self.cfg.env, "ee_goal_obs_mode", "command")
        if ee_goal_obs_mode is None:
            ee_goal_obs_mode = "command"

        if ee_goal_obs_mode == "command":
            ee_goal_local_cart = self.curr_ee_goal_cart
        elif ee_goal_obs_mode == "arm_base_target":
            arm_base_name = None
            if hasattr(self.cfg, "asset"):
                arm_base_name = getattr(self.cfg.asset, "arm_waist_name", None)
            if arm_base_name is None:
                arm_base_name = "joint1"

            if (
                hasattr(self, "body_names_to_idx")
                and arm_base_name in self.body_names_to_idx
                and hasattr(self.robot.data, "body_state_w")
                and hasattr(self, "curr_ee_goal_cart_world")
            ):
                arm_base_idx = self.body_names_to_idx[arm_base_name]
                arm_base_pos = self.robot.data.body_state_w[:, arm_base_idx, :3]
                ee_goal_local_cart = quat_rotate_inverse(
                    self.base_quat,
                    self.curr_ee_goal_cart_world - arm_base_pos,
                )
            else:
                ee_goal_local_cart = self.curr_ee_goal_cart
        else:
            raise ValueError(f"Unsupported ee_goal_obs_mode: {ee_goal_obs_mode}")

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

        return {"policy": self.obs_buf}

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies, walking_mask = self._get_gait_frequencies()
            phases = 0.5
            offsets = 0
            bounds = 0
            durations = 0.5
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
            self.gait_indices[~walking_mask] = 0

            canonical_foot_indices = {
                "FL_foot": self.gait_indices + phases + offsets + bounds,
                "FR_foot": self.gait_indices + offsets,
                "RL_foot": self.gait_indices + bounds,
                "RR_foot": self.gait_indices + phases,
            }
            policy_foot_names = list(self.cfg.asset.policy_foot_names)
            raw_foot_indices = {
                foot_name: torch.remainder(canonical_foot_indices[foot_name], 1.0)
                for foot_name in policy_foot_names
            }

            self.foot_indices = torch.cat(
                [raw_foot_indices[foot_name].unsqueeze(1) for foot_name in policy_foot_names],
                dim=1,
            )

            shaped_foot_indices = {}
            for foot_name, base_indices in canonical_foot_indices.items():
                idxs = base_indices.clone()
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                            0.5 / (1 - durations))
                shaped_foot_indices[foot_name] = idxs

            for i, foot_name in enumerate(policy_foot_names):
                idxs = shaped_foot_indices[foot_name]
                self.clock_inputs[:, i] = torch.sin(2 * np.pi * idxs)
                self.doubletime_clock_inputs[:, i] = torch.sin(4 * np.pi * idxs)
                self.halftime_clock_inputs[:, i] = torch.sin(np.pi * idxs)

            # def _compute_smoothing_multiplier(idxs):
            #     phase = torch.remainder(idxs, 1.0)
            #     return (
            #         smoothing_cdf_start(phase) * (1 - smoothing_cdf_start(phase - 0.5))
            #         + smoothing_cdf_start(phase - 1) * (1 - smoothing_cdf_start(phase - 1.5))
            #     )

            # # von mises distribution
            # kappa = self.cfg.rewards.kappa_gait_probs
            # smoothing_cdf_start = torch.distributions.normal.Normal(0,
            #                                                         kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            # smoothing_multipliers = {
            #     foot_name: _compute_smoothing_multiplier(shaped_foot_indices[foot_name])
            #     for foot_name in policy_foot_names
            # }

            # for i, foot_name in enumerate(policy_foot_names):
            #     self.desired_contact_states[:, i] = smoothing_multipliers[foot_name]

    def _get_gait_frequencies(self):
        min_frequency = float(self.cfg.env.gait_frequency_min)
        max_frequency = float(self.cfg.env.gait_frequency_max)
        if max_frequency < min_frequency:
            min_frequency, max_frequency = max_frequency, min_frequency

        lin_vel_ref = max(float(self.cfg.env.gait_frequency_lin_vel_ref), 1e-6)
        ang_vel_ref = max(float(self.cfg.env.gait_frequency_ang_vel_ref), 1e-6)
        ang_vel_weight = max(float(self.cfg.env.gait_frequency_ang_vel_weight), 0.0)

        lin_cmd_level = torch.norm(self.commands[:, :2], dim=1) / lin_vel_ref
        yaw_cmd_level = torch.abs(self.commands[:, 2]) / ang_vel_ref
        gait_level = torch.clamp(lin_cmd_level + ang_vel_weight * yaw_cmd_level, 0.0, 1.0)

        frequencies = min_frequency + (max_frequency - min_frequency) * gait_level
        walking_mask = self._get_walking_cmd_mask()
        frequencies = torch.where(walking_mask, frequencies, torch.zeros_like(frequencies))
        self.gait_frequencies[:] = frequencies
        return frequencies, walking_mask
    
    def _get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_x_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
        
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
        self.commands[:] = 0.0
        self.commands[:, 0] = 1.0
        self.commands[:, 2] = 0.0

    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :] *= (torch.logical_or(torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip, torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)).unsqueeze(1)

    # legacy API conveniences used by old scripts
    def get_observations(self):
        # return self._get_observations()["policy"]
        return self._get_observations()

    def get_privileged_observations(self):
        return None

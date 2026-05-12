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
        self.dt = self.cfg.sim.dt * self.cfg.decimation
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
        ############################################                

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
        clip = float(self.cfg.control.clip_actions)
        self.actions = torch.clip(actions, -clip, clip).to(self.device)
        self.actions[:, 12:] = 0.0
        if self.actions.shape[-1] != self.num_actions:
            raise RuntimeError(f"Expected action dim {self.num_actions}, got {self.actions.shape[-1]}")
        self.torques = self._compute_torques(self.actions)

    def _apply_action(self):
        # 1. Hold arm/gripper by position targets, matching old set_dof_position_target_tensor().
        arm_gripper_ids = torch.cat([self.arm_joint_ids, self.gripper_joint_ids])
        if arm_gripper_ids.numel() > 0:
            self.robot.set_joint_position_target(
                self.default_dof_pos[:, arm_gripper_ids],
                joint_ids=arm_gripper_ids,
            )
        # self.target_pos = self.default_dof_pos.clone()            
        self.robot.set_joint_position_target(self.target_pos)            
                    
        self.robot.set_joint_effort_target(self.torques)          

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        # torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        # scaled = actions * float(self.cfg.control.action_scale)
        # joint_ids = self.policy_joint_ids
        # if self.cfg.control.control_type == "P":
        #     target = self.default_dof_pos[:, joint_ids] + scaled[:, : joint_ids.numel()]
        #     torques[:, joint_ids] = (
        #         self.p_gains[joint_ids] * (target - self.dof_pos[:, joint_ids])
        #         - self.d_gains[joint_ids] * self.dof_vel[:, joint_ids]
        #     )
        # elif self.cfg.control.control_type == "T":
        #     torques[:, joint_ids] = scaled[:, : joint_ids.numel()]
        # else:
        #     raise NotImplementedError("Only P and T control are implemented in this IsaacLab skeleton.")
        # effort_limits = self.robot.data.soft_joint_vel_limits * 0.0 + 600.0
        # return torch.clamp(torques, -effort_limits, effort_limits)
        # scaled = actions * self.action_scale_tensor.unsqueeze(0)
        scaled = actions * self.action_scale_tensor.unsqueeze(0)
        target_pos = self.default_dof_pos.clone()
        joint_ids = self.policy_joint_ids

        # policy action controls 18 policy joints in sim joint order
        target_pos[:, joint_ids] = self.default_dof_pos[:, joint_ids] + scaled[:, : joint_ids.numel()]

        # keep this for IsaacLab position drive
        self.target_pos = target_pos

        if self.cfg.control.control_type == "P":
            torques = (
                self.p_gains.unsqueeze(0) * (target_pos - self.dof_pos)
                - self.d_gains.unsqueeze(0) * self.dof_vel
            )
        elif self.cfg.control.control_type == "T":
            torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
            torques[:, joint_ids] = scaled[:, : joint_ids.numel()]
        else:
            raise NotImplementedError

        # if you want old ManipLoco closer: arm torque zero, arm held by position target
        if hasattr(self, "arm_joint_ids"):
            torques[:, self.arm_joint_ids] = 0.0
        if hasattr(self, "gripper_joint_ids"):
            torques[:, self.gripper_joint_ids] = 0.0

        return torch.clamp(torques, -600.0, 600.0)

    def _get_observations(self):
        obs = torch.cat((
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.commands[:, :3],
            (self.dof_pos[:, self.policy_joint_ids] - self.default_dof_pos[:, self.policy_joint_ids]),
            self.dof_vel[:, self.policy_joint_ids] * 0.05,
            self.actions,
        ), dim=-1)
        if obs.shape[-1] < self.num_obs:
            obs = torch.cat((obs, torch.zeros(self.num_envs, self.num_obs - obs.shape[-1], device=self.device)), dim=-1)
        elif obs.shape[-1] > self.num_obs:
            obs = obs[:, :self.num_obs]
        obs = torch.clip(obs, -self.cfg.control.clip_observations, self.cfg.control.clip_observations)
        self.last_actions[:] = self.actions
        return {"policy": obs}

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
        return self._get_observations()["policy"]

    def get_privileged_observations(self):
        return None

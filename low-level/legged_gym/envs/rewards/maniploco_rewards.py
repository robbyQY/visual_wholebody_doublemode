import torch
from isaacgym.torch_utils import *


class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _get_base_height(self):
        env_origin_z = getattr(self.env, "env_origins", None)
        if env_origin_z is None:
            return self.env.root_states[:, 2]
        return self.env.root_states[:, 2] - env_origin_z[:, 2]

    def _get_height_conditioned_leg_reference(self):
        default = self.env.default_dof_pos
        if default.dim() == 1:
            stand_ref = default[:12]
        else:
            stand_ref = default[0, :12]
        crouch_ref = stand_ref.clone()
        tiptoe_ref = stand_ref.clone()
        for dof_idx, dof_name in enumerate(self.env.dof_names[:12]):
            if "hip" in dof_name:
                crouch_ref[dof_idx] += self.env.cfg.rewards.crouch_hip_delta
                tiptoe_ref[dof_idx] += self.env.cfg.rewards.tiptoe_hip_delta
            elif "thigh" in dof_name:
                crouch_ref[dof_idx] += self.env.cfg.rewards.crouch_thigh_delta
                tiptoe_ref[dof_idx] += self.env.cfg.rewards.tiptoe_thigh_delta
            elif "calf" in dof_name:
                crouch_ref[dof_idx] += self.env.cfg.rewards.crouch_calf_delta
                tiptoe_ref[dof_idx] += self.env.cfg.rewards.tiptoe_calf_delta

        base_height = self._get_base_height()
        stand_height = float(self.env.cfg.rewards.base_height_target)
        crouch_height = float(self.env.cfg.rewards.base_height_target_min)
        crouch_span = max(stand_height - crouch_height, 1e-6)
        crouch_alpha = ((stand_height - base_height) / crouch_span).clamp(0.0, 1.0)
        lower_ref = torch.lerp(
            stand_ref.unsqueeze(0).expand(self.env.num_envs, -1),
            crouch_ref.unsqueeze(0).expand(self.env.num_envs, -1),
            crouch_alpha.unsqueeze(1),
        )

        tiptoe_height = float(self.env.cfg.rewards.base_height_target_max)
        tiptoe_span = max(tiptoe_height - stand_height, 1e-6)
        tiptoe_alpha = ((base_height - stand_height) / tiptoe_span).clamp(0.0, 1.0)
        upper_ref = torch.lerp(
            stand_ref.unsqueeze(0).expand(self.env.num_envs, -1),
            tiptoe_ref.unsqueeze(0).expand(self.env.num_envs, -1),
            tiptoe_alpha.unsqueeze(1),
        )
        return torch.where((base_height <= stand_height).unsqueeze(1), lower_ref, upper_ref)

    def _get_leg_posture_tracking_reward(self):
        ref = self._get_height_conditioned_leg_reference()
        dof_error = torch.sum(torch.abs(self.env.dof_pos[:, :12] - ref), dim=1)
        reward = torch.exp(-dof_error * self.env.cfg.rewards.leg_posture_exp_scale)
        metric = torch.rad2deg(dof_error / 12.0)
        return reward, metric

    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.env.get_goal_reference_quat(), self.env.ee_pos - self.env.get_ee_goal_spherical_center())
        ee_pos_error = cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere
        ee_pos_error[:, 2] = torch_wrap_to_pi_minuspi(ee_pos_error[:, 2])
        ee_pos_error = torch.sum(torch.abs(ee_pos_error) * self.env.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.transform_goal_local_to_world(self.env.get_goal_center_offset_local() + self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), torch.rad2deg(orn_err)

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, 12:-self.env.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 12:-self.env.cfg.env.num_gripper_joints]), dim=1)
        return energy, energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), torch.rad2deg(orn_err)

    # -------------Existing leg reward functions----------------

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2, torch.sqrt(action_l2)

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim=1))
        return energy, energy

    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, :12] ** 2, dim=1)
        return action_l2, torch.sqrt(action_l2)

    def _reward_leg_energy(self):
        energy = torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim=1)
        return energy, torch.abs(energy)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma), torch.sqrt(lin_vel_error)

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0 * error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return -error + torch.abs(self.env.commands[:, 2]), error

    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, torch.sqrt(squared_error)

    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = (self.env.commands[:, 2] - self.env.base_lin_vel[:, 2]) ** 2
        return squared_error, torch.sqrt(squared_error)

    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, torch.sqrt(foot_contacts_z)

    def _reward_torques(self):
        torque = torch.sum(torch.square(self.env.torques), dim=1)
        return torque, torch.sqrt(torque)

    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, torch.sqrt(energy)

    def _reward_tracking_lin_vel_y(self):
        cmd = self.env.commands[:, 1].clone()
        lin_vel_y_error = torch.square(cmd - self.env.base_lin_vel[:, 1])
        rew = torch.exp(-lin_vel_y_error/self.env.cfg.rewards.tracking_sigma)
        return rew, torch.sqrt(lin_vel_y_error)

    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, torch.sqrt(rew)

    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, torch.sqrt(rew)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma), torch.sqrt(ang_vel_error)

    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, :12], dim=1))
        return abs_sum_work, abs_sum_work

    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return rew, torch.sqrt(rew)

    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, :12], dim=1)
        return action_rate, torch.sqrt(action_rate)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        rew = torch.sum(out_of_limits[:, :12], dim=1)
        return rew, torch.rad2deg(rew)

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, :12], dim=1)
        return rew, torch.sqrt(rew)

    def _reward_collision(self):
        rew = torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
        return rew, rew

    def _reward_stand_still(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error * 0.05)
        walking_mask = self.env._get_walking_cmd_mask()
        rew[walking_mask] = 0.
        metric = torch.rad2deg(dof_error / 12.0)
        metric[walking_mask] = 0.
        return rew, metric

    def _reward_stand_still_flexible(self):
        rew, metric = self._get_leg_posture_tracking_reward()
        walking_mask = self.env._get_walking_cmd_mask()
        rew[walking_mask] = 0.
        metric[walking_mask] = 0.
        return rew, metric

    def _reward_walking_dof(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error * 0.05)
        walking_mask = self.env._get_walking_cmd_mask()
        rew[~walking_mask] = 0.
        metric = torch.rad2deg(dof_error / 12.0)
        metric[~walking_mask] = 0.
        return rew, metric

    def _reward_walking_dof_flexible(self):
        rew, metric = self._get_leg_posture_tracking_reward()
        walking_mask = self.env._get_walking_cmd_mask()
        rew[~walking_mask] = 0.
        metric[~walking_mask] = 0.
        return rew, metric

    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        return rew, torch.rad2deg(torch.sqrt(rew / len(self.env.hip_indices)))

    def _reward_hip_pos_flexible(self):
        ref = self._get_height_conditioned_leg_reference()
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - ref[:, self.env.hip_indices]), dim=1)
        return rew, torch.rad2deg(torch.sqrt(rew / len(self.env.hip_indices)))

    def _reward_feet_jerk(self):
        if not hasattr(self.env, "last_contact_forces"):
            result = torch.zeros(self.env.num_envs).to(self.env.device)
            metric = result.clone()
        else:
            force_delta = torch.norm(self.env.force_sensor_tensor - self.env.last_contact_forces, dim=-1)
            result = torch.sum(force_delta, dim=-1)
            metric = result / force_delta.shape[1]
        self.env.last_contact_forces = self.env.force_sensor_tensor.clone()
        result[self.env.episode_length_buf < 50] = 0.
        metric[self.env.episode_length_buf < 50] = 0.
        return result, metric

    def _reward_alive(self):
        return 1., 1.

    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.env.rigid_body_state[:, self.env.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        metric = rew / self.env.foot_contacts_from_sensor.float().sum(dim=1).clamp(min=1.0)
        return rew, metric

    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2. / self.env.dt).type(torch.float)
        forces = torch.sum((torch.norm(self.env.force_sensor_tensor, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0), dim=-1)
        rew = reset_flag * forces
        return rew, rew

    def _reward_orientation(self):
        error = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        return error, torch.rad2deg(torch.asin(torch.sqrt(error.clamp(min=0.0, max=1.0))))

    def _reward_roll(self):
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, torch.rad2deg(error)

    def _reward_pitch(self):
        pitch = self.env._get_body_orientation()[:, 1]
        error = torch.abs(pitch)
        return error, torch.rad2deg(error)

    def _reward_base_height(self):
        base_height = self._get_base_height()
        error = torch.abs(base_height - self.env.cfg.rewards.base_height_target)
        return error, error

    def _reward_base_height_nominal(self):
        base_height = self._get_base_height()
        error = torch.abs(base_height - self.env.cfg.rewards.base_height_target)
        return torch.square(error), error

    def _reward_base_height_band(self):
        base_height = self._get_base_height()
        low = float(self.env.cfg.rewards.base_height_target_min)
        high = float(self.env.cfg.rewards.base_height_target_max)
        below = (low - base_height).clip(min=0.0)
        above = (base_height - high).clip(min=0.0)
        error = below + above
        return error, error

    def _reward_orientation_walking(self):
        reward, metric = self._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_orientation_standing(self):
        reward, metric = self._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_torques_walking(self):
        reward, metric = self._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward, metric = self._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_energy_square_walking(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_energy_square_standing(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward, metric = self._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward, metric = self._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_dof_default_pos(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error * 0.05)
        return rew, torch.rad2deg(dof_error / 12.0)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        return dof_error, torch.rad2deg(torch.sqrt(dof_error / 12.0))

    def _reward_tracking_lin_vel_max(self):
        cmd_x = self.env.commands[:, 0]
        vel_x = self.env.base_lin_vel[:, 0]
        abs_cmd_x = torch.abs(cmd_x)
        tracked_vel_x = torch.where(cmd_x > 0, torch.minimum(vel_x, cmd_x), torch.minimum(-vel_x, -cmd_x))
        rew = tracked_vel_x / (abs_cmd_x + 1e-5)
        zero_cmd_indices = abs_cmd_x < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(vel_x))[zero_cmd_indices]
        metric = (abs_cmd_x - tracked_vel_x).clamp(min=0.0)
        metric[zero_cmd_indices] = torch.abs(vel_x)[zero_cmd_indices]
        return rew, metric

    def _reward_penalty_lin_vel_y(self):
        rew = torch.abs(self.env.base_lin_vel[:, 1])
        rot_indices = torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.ang_vel_yaw_clip
        rew[rot_indices] = 0.
        return rew, rew

    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0, 0
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states
        transition_lower = getattr(self.env.cfg.rewards, "gait_transition_lower", 0.2)
        stable_swing = desired_contact < transition_lower
        active_foot_forces = foot_forces * stable_swing.float()
        reward = torch.sum(1 - torch.exp(-active_foot_forces ** 2 / self.env.cfg.rewards.gait_force_sigma), dim=1)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        reward[cmd_stop_flag] = 0
        metric = torch.sum(active_foot_forces, dim=1) / torch.sum(stable_swing.float(), dim=1).clamp(min=1.0)
        metric[cmd_stop_flag] = 0
        return reward / 4, metric

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0, 0
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        transition_upper = getattr(self.env.cfg.rewards, "gait_transition_upper", 0.8)
        stable_stance = desired_contact > transition_upper
        active_foot_velocities = foot_velocities * stable_stance.float()
        reward = torch.sum(1 - torch.exp(-active_foot_velocities ** 2 / self.env.cfg.rewards.gait_vel_sigma), dim=1)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        reward[cmd_stop_flag] = 0
        metric = torch.sum(active_foot_velocities, dim=1) / torch.sum(stable_stance.float(), dim=1).clamp(min=1.0)
        metric[cmd_stop_flag] = 0
        return reward / 4, metric

    def _reward_feet_height(self):
        feet_height_tracking = self.env.cfg.rewards.feet_height_target
        if self.env.cfg.rewards.feet_height_allfeet:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices, 2]
        else:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices[:2], 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - feet_height_tracking, max=0)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew, rew

    def _reward_feet_air_time(self):
        first_contact = (self.env.feet_air_time > 0.) * self.env.foot_contacts_from_sensor
        self.env.feet_air_time += self.env.dt
        if self.env.cfg.rewards.feet_airtime_allfeet:
            first_contact_float = first_contact.float()
            air_time_at_contact = self.env.feet_air_time * first_contact_float
        else:
            first_contact_float = first_contact[:, :2].float()
            air_time_at_contact = self.env.feet_air_time[:, :2] * first_contact_float
        first_contact_count = first_contact_float.sum(dim=1)
        rew_airTime = torch.sum(air_time_at_contact - 0.5 * first_contact_float, dim=1)
        rew_airTime *= self.env._get_walking_cmd_mask()
        metric = torch.where(first_contact_count > 0, torch.sum(air_time_at_contact, dim=1) / first_contact_count, torch.zeros_like(rew_airTime))
        metric *= self.env._get_walking_cmd_mask()
        self.env.feet_air_time *= ~self.env.foot_contacts_from_sensor
        return rew_airTime, metric

    # -------------robot_lab Unitree B2 reproduction rewards----------------

    def _robotlab_upright_gate(self):
        return torch.clamp(-self.env.projected_gravity[:, 2], 0.0, 0.7) / 0.7

    def _robotlab_command_norm(self):
        return torch.linalg.norm(self.env.commands[:, :3], dim=1)

    def _robotlab_default_dof_pos(self):
        default = self.env.default_dof_pos
        if default.dim() == 1:
            default = default.unsqueeze(0)
        if default.shape[0] == 1:
            default = default.expand(self.env.num_envs, -1)
        return default

    def _robotlab_dof_indices(self, joint_names):
        if not hasattr(self.env, "_robotlab_dof_name_to_index"):
            self.env._robotlab_dof_name_to_index = {name: i for i, name in enumerate(self.env.dof_names)}
        return torch.tensor([self.env._robotlab_dof_name_to_index[name] for name in joint_names], device=self.env.device, dtype=torch.long)

    def _robotlab_nonfoot_body_indices(self):
        if not hasattr(self.env, "_robotlab_nonfoot_body_indices"):
            num_bodies = self.env.contact_forces.shape[1]
            foot_ids = set(int(i) for i in self.env.feet_indices.detach().cpu().tolist())
            nonfoot = [i for i in range(num_bodies) if i not in foot_ids]
            self.env._robotlab_nonfoot_body_indices = torch.tensor(nonfoot, device=self.env.device, dtype=torch.long)
        return self.env._robotlab_nonfoot_body_indices

    def _robotlab_first_contact_from_sensor(self):
        contacts = self.env.foot_contacts_from_sensor.bool()
        if not hasattr(self.env, "_robotlab_last_foot_contacts") or self.env._robotlab_last_foot_contacts.shape != contacts.shape:
            self.env._robotlab_last_foot_contacts = torch.zeros_like(contacts)
        first_contact = contacts & (~self.env._robotlab_last_foot_contacts)
        self.env._robotlab_last_foot_contacts = contacts.clone()
        return first_contact

    def _reward_robotlab_track_lin_vel_xy_exp(self):
        std = float(getattr(self.env.cfg.rewards, "robotlab_tracking_std", 0.5))
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error / (std ** 2)) * self._robotlab_upright_gate()
        return reward, torch.sqrt(lin_vel_error)

    def _reward_robotlab_track_ang_vel_z_exp(self):
        std = float(getattr(self.env.cfg.rewards, "robotlab_tracking_std", 0.5))
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error / (std ** 2)) * self._robotlab_upright_gate()
        return reward, torch.sqrt(ang_vel_error)

    def _reward_robotlab_upward(self):
        reward = torch.square(1.0 - self.env.projected_gravity[:, 2])
        return reward, reward

    def _reward_robotlab_lin_vel_z_l2(self):
        reward = torch.square(self.env.base_lin_vel[:, 2]) * self._robotlab_upright_gate()
        return reward, torch.sqrt(reward)

    def _reward_robotlab_ang_vel_xy_l2(self):
        reward = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1) * self._robotlab_upright_gate()
        return reward, torch.sqrt(reward)

    def _reward_robotlab_joint_torques_l2(self):
        reward = torch.sum(torch.square(self.env.torques[:, :12]), dim=1)
        return reward, torch.sqrt(reward)

    def _reward_robotlab_joint_acc_l2(self):
        reward = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return reward, torch.sqrt(reward)

    def _reward_robotlab_joint_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.0)
        reward = torch.sum(out_of_limits[:, :12], dim=1)
        return reward, torch.rad2deg(reward)

    def _reward_robotlab_joint_power(self):
        reward = torch.sum(torch.abs(self.env.dof_vel[:, :12] * self.env.torques[:, :12]), dim=1)
        return reward, reward

    def _reward_robotlab_stand_still(self):
        default = self._robotlab_default_dof_pos()
        reward = torch.sum(torch.abs(self.env.dof_pos[:, :12] - default[:, :12]), dim=1)
        reward = reward * (self._robotlab_command_norm() < 0.1).float()
        reward = reward * self._robotlab_upright_gate()
        return reward, torch.rad2deg(reward / 12.0)

    def _reward_robotlab_joint_pos_penalty(self):
        default = self._robotlab_default_dof_pos()
        running_reward = torch.linalg.norm(self.env.dof_pos[:, :12] - default[:, :12], dim=1)
        cmd = self._robotlab_command_norm()
        body_vel = torch.linalg.norm(self.env.base_lin_vel[:, :2], dim=1)
        stand_still_scale = float(getattr(self.env.cfg.rewards, "robotlab_stand_still_scale", 5.0))
        velocity_threshold = float(getattr(self.env.cfg.rewards, "robotlab_velocity_threshold", 0.5))
        command_threshold = float(getattr(self.env.cfg.rewards, "robotlab_command_threshold", 0.1))
        reward = torch.where(torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold), running_reward, stand_still_scale * running_reward)
        reward = reward * self._robotlab_upright_gate()
        return reward, torch.rad2deg(running_reward / 12.0)

    def _reward_robotlab_joint_mirror(self):
        pair_names = [
            (["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"], ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]),
            (["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"], ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]),
        ]
        reward = torch.zeros(self.env.num_envs, device=self.env.device)
        for a_names, b_names in pair_names:
            a_idx = self._robotlab_dof_indices(a_names)
            b_idx = self._robotlab_dof_indices(b_names)
            reward += torch.sum(torch.square(self.env.dof_pos[:, a_idx] - self.env.dof_pos[:, b_idx]), dim=1)
        reward = reward / len(pair_names)
        reward = reward * self._robotlab_upright_gate()
        return reward, torch.sqrt(reward.clamp(min=0.0))

    def _reward_robotlab_action_rate_l2(self):
        reward = torch.sum(torch.square(self.env.last_actions[:, :12] - self.env.actions[:, :12]), dim=1)
        return reward, torch.sqrt(reward)

    def _reward_robotlab_undesired_contacts(self):
        threshold = float(getattr(self.env.cfg.rewards, "robotlab_undesired_contacts_threshold", 1.0))
        nonfoot_ids = self._robotlab_nonfoot_body_indices()
        force_norm = torch.norm(self.env.contact_forces[:, nonfoot_ids, :], dim=-1)
        reward = torch.sum((force_norm > threshold).float(), dim=1) * self._robotlab_upright_gate()
        return reward, reward

    def _reward_robotlab_contact_forces(self):
        threshold = float(getattr(self.env.cfg.rewards, "robotlab_contact_force_threshold", 100.0))
        forces = torch.norm(self.env.force_sensor_tensor, dim=-1)
        reward = torch.sum((forces - threshold).clamp(min=0.0), dim=-1)
        return reward, reward

    def _reward_robotlab_feet_height_body(self):
        target_height = float(getattr(self.env.cfg.rewards, "robotlab_feet_height_body_target", -0.4))
        tanh_mult = float(getattr(self.env.cfg.rewards, "robotlab_feet_height_tanh_mult", 2.0))
        foot_pos_w = self.env.rigid_body_state[:, self.env.feet_indices, 0:3]
        foot_vel_w = self.env.rigid_body_state[:, self.env.feet_indices, 7:10]
        root_pos_w = self.env.root_states[:, 0:3]
        root_vel_w = self.env.root_states[:, 7:10]
        root_quat_w = self.env.root_states[:, 3:7]
        n_feet = foot_pos_w.shape[1]
        rel_pos = (foot_pos_w - root_pos_w.unsqueeze(1)).reshape(-1, 3)
        rel_vel = (foot_vel_w - root_vel_w.unsqueeze(1)).reshape(-1, 3)
        quat = root_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
        foot_pos_b = quat_rotate_inverse(quat, rel_pos).view(self.env.num_envs, n_feet, 3)
        foot_vel_b = quat_rotate_inverse(quat, rel_vel).view(self.env.num_envs, n_feet, 3)
        foot_z_target_error = torch.square(foot_pos_b[:, :, 2] - target_height)
        foot_velocity_tanh = torch.tanh(tanh_mult * torch.linalg.norm(foot_vel_b[:, :, :2], dim=2))
        reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
        reward = reward * (self._robotlab_command_norm() > 0.1).float()
        reward = reward * self._robotlab_upright_gate()
        return reward, reward

    def _robotlab_clock_gait_weights(self):
        # robot_lab-clock：根据现有 desired_contact_states 生成 swing / stance 权重。
        # desired_contact_states 接近 0 表示该脚处于 swing，相应地惩罚触地力；
        # desired_contact_states 接近 1 表示该脚处于 stance，相应地惩罚滑动和接触不足；
        # 中间过渡区不强惩罚，避免相位切换瞬间 reward 过硬。
        desired_contact = self.env.desired_contact_states.clamp(0.0, 1.0)

        transition_lower = getattr(self.env.cfg.rewards, "gait_transition_lower", 0.1)
        transition_upper = getattr(self.env.cfg.rewards, "gait_transition_upper", 0.9)

        swing_weight = (
            (transition_lower - desired_contact)
            / max(transition_lower, 1e-6)
        ).clamp(0.0, 1.0)

        stance_weight = (
            (desired_contact - transition_upper)
            / max(1.0 - transition_upper, 1e-6)
        ).clamp(0.0, 1.0)

        return swing_weight, stance_weight

    def _reward_robotlab_clock_swing_force(self):
        # robot_lab-clock：跟随 clock_inputs / desired_contact_states。
        # 当某只脚处于 swing 相位时，惩罚它产生接触力。
        # 静止时不使用 clock gait，避免静止四腿着地和 clock swing 相位冲突。
        if not self.env.cfg.env.observe_gait_commands:
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        walking_mask = self.env._get_walking_cmd_mask()
        if not torch.any(walking_mask):
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        swing_weight, _ = self._robotlab_clock_gait_weights()

        foot_forces = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices, :],
            dim=-1,
        )

        gait_force_sigma = float(self.env.cfg.rewards.gait_force_sigma)

        penalty = swing_weight * (
            1.0 - torch.exp(-foot_forces ** 2 / gait_force_sigma)
        )

        rew = torch.sum(penalty, dim=1) / 4.0
        metric = torch.sum(swing_weight * foot_forces, dim=1) / torch.sum(
            swing_weight,
            dim=1,
        ).clamp(min=1.0)

        rew[~walking_mask] = 0.0
        metric[~walking_mask] = 0.0

        return rew, metric

    def _reward_robotlab_clock_stance_vel(self):
        # robot_lab-clock：跟随 clock_inputs / desired_contact_states。
        # 当某只脚处于 stance 相位时，惩罚它在地面上滑动。
        # 静止时不使用该项，静止四脚接触由 robotlab_feet_contact_without_cmd 处理。
        if not self.env.cfg.env.observe_gait_commands:
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        walking_mask = self.env._get_walking_cmd_mask()
        if not torch.any(walking_mask):
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        _, stance_weight = self._robotlab_clock_gait_weights()

        foot_velocities = torch.norm(
            self.env.foot_velocities,
            dim=2,
        ).view(self.env.num_envs, -1)

        gait_vel_sigma = float(self.env.cfg.rewards.gait_vel_sigma)

        penalty = stance_weight * (
            1.0 - torch.exp(-foot_velocities ** 2 / gait_vel_sigma)
        )

        rew = torch.sum(penalty, dim=1) / 4.0
        metric = torch.sum(stance_weight * foot_velocities, dim=1) / torch.sum(
            stance_weight,
            dim=1,
        ).clamp(min=1.0)

        rew[~walking_mask] = 0.0
        metric[~walking_mask] = 0.0

        return rew, metric

    def _reward_robotlab_clock_stance_contact(self):
        # robot_lab-clock：跟随 clock_inputs / desired_contact_states。
        # 当某只脚处于 stance 相位时，轻微惩罚它没有形成足够接触力。
        # 该项只作为辅助，权重应小于 swing_force 和 stance_vel。
        if not self.env.cfg.env.observe_gait_commands:
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        walking_mask = self.env._get_walking_cmd_mask()
        if not torch.any(walking_mask):
            zero = torch.zeros(self.env.num_envs, device=self.env.device)
            return zero, zero

        _, stance_weight = self._robotlab_clock_gait_weights()

        foot_forces = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices, :],
            dim=-1,
        )

        gait_force_sigma = float(self.env.cfg.rewards.gait_force_sigma)

        penalty = stance_weight * torch.exp(
            -foot_forces ** 2 / gait_force_sigma
        )

        rew = torch.sum(penalty, dim=1) / 4.0
        metric = torch.sum(stance_weight * foot_forces, dim=1) / torch.sum(
            stance_weight,
            dim=1,
        ).clamp(min=1.0)

        rew[~walking_mask] = 0.0
        metric[~walking_mask] = 0.0

        return rew, metric

    def _reward_robotlab_feet_contact_without_cmd(self):
        # robot_lab：无运动命令时奖励足端保持接触。
        # 这里不使用 first-contact 事件，而是奖励当前四脚着地状态。
        # 原因是静止站立时，四只脚稳定接触后不会持续产生 first-contact，
        # 如果只奖励 first-contact，静止站稳后 reward 反而没有持续信号。
        foot_contacts = self.env.foot_contacts_from_sensor.float()

        rew = torch.sum(foot_contacts, dim=1) / foot_contacts.shape[1]

        standing_mask = ~self.env._get_walking_cmd_mask()
        rew[~standing_mask] = 0.0

        rew = rew * self._robotlab_upright_gate()
        metric = rew

        return rew, metric

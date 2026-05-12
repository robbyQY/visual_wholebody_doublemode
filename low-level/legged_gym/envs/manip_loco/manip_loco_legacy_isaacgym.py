# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, torch_rand_sqrt_float, wrap_to_pi

import torch
import torch.distributed as dist
from typing import Tuple, Dict

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.terrain import Terrain, Terrain_Perlin
from rsl_rl.utils import resolve_schedule_value
from .manip_loco_base_config import ManipLocoRoughCfg

import sys

class ManipLoco(LeggedRobot):
    name = None
    cfg: ManipLocoRoughCfg

    def __init__(self, cfg, *args, **kwargs):
        is_main_process = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        if cfg.goal_ee.sphere_center.mixed_height_reference:
            cfg.env.num_proprio += 1
            cfg.env.num_observations = cfg.env.num_proprio * (cfg.env.history_len+1) + cfg.env.num_priv
            self.num_obs = cfg.env.num_observations
        if cfg.env.observe_gait_commands:
            if is_main_process:
                print("||||||||||Observe gait commands!")
            cfg.env.num_proprio += 5 # gait_indices=1, clock_phase=4
            cfg.env.num_observations = cfg.env.num_proprio * (cfg.env.history_len+1) + cfg.env.num_priv
            self.num_obs = cfg.env.num_observations
        self.stand_by = cfg.env.stand_by
        super().__init__(cfg, *args, **kwargs)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions[:, 12:] = 0.
        actions = self._policy_to_env_all(actions)
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        if self.consume_manual_reset_request():
            self.reset_all_envs(start=True)
            actions.zero_()
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
            # actions = self.action_history_buf[:, -self.action_delay - 1] # delay for 1/50=20ms
        action_delay_mode = self.cfg.env.action_delay_mode
        if action_delay_mode == "undelayed":
            actions = self.action_history_buf[:, -1]
        elif action_delay_mode == "delayed":
            actions = self.action_history_buf[:, -2]
        else:
            if self.global_steps < 10000 * 24:
                actions = self.action_history_buf[:, -1]
            else:
                actions = self.action_history_buf[:, -2]

        self.actions = actions.clone()
        arm_pos_targets = self._get_arm_pos_targets()
        all_pos_targets = torch.zeros_like(self.dof_pos)
        all_pos_targets[:, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints] = arm_pos_targets
        all_pos_targets[:, -self.cfg.env.num_gripper_joints:] = self.gripper_pos_targets

        for t in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.global_steps += 1
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras

    def _get_arm_pos_targets(self):
        if self.cfg.env.teleop_mode and self.teleop_arm_control_mode == "joint":
            return self.teleop_arm_joint_pos_targets

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        drot = orientation_error(self.ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], -1).unsqueeze(-1)
        return self._control_ik(dpose) + self.dof_pos[:, self.arm_dof_start_idx:self.arm_dof_end_idx]

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler[:] = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 1.5    

        self._post_physics_step_callback()
        
        # update ee goal
        self._update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids, start=False)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_ee_goal_curr()
            self._draw_ee_goal_traj()
            self._draw_collision_bbox()

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew, metric = self.reward_functions[i]()
            rew = rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric * self.dt
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew, metric = self._reward_termination()
            rew = rew * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.episode_metric_sums["termination"] += metric * self.dt
        
        self.rew_buf /= 100

        self.arm_rew_buf[:] = 0.
        for i in range(len(self.arm_reward_functions)):
            name = self.arm_reward_names[i]
            rew, metric = self.arm_reward_functions[i]()
            rew = rew * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric * self.dt
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.)
        # add termination reward after clipping
        if "arm_termination" in self.arm_reward_scales:
            rew, metric = self._reward_termination()
            rew = rew * self.arm_reward_scales["arm_termination"]
            self.arm_rew_buf += rew
            self.episode_sums["arm_termination"] += rew
            self.episode_metric_sums["arm_termination"] += metric * self.dt

        self.arm_rew_buf /= 100

    def compute_observations(self):
        """Computes policy observations.

        `ee_goal_local_cart` is always expressed in the robot body frame.
        It is either the commanded offset directly, or the world target re-expressed relative to the arm base.
        """
        if self.cfg.env.teleop_mode:
            self._update_effective_teleop_inputs()
        ee_goal_obs_mode = self.cfg.env.ee_goal_obs_mode
        if ee_goal_obs_mode == "command":
            # Match the goal-sampling command semantics used by newer checkpoints.
            ee_goal_local_cart = self.curr_ee_goal_cart
        elif ee_goal_obs_mode == "arm_base_target":
            # Legacy checkpoints observed the world target relative to the arm base.
            arm_base_pos = self._get_arm_base_world_pos()
            ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)
        else:
            raise ValueError(f"Unsupported ee_goal_obs_mode: {ee_goal_obs_mode}")
        if self.stand_by:
            self.commands[:] = 0.

        obs_terms = [self._get_body_orientation(),  # dim 2
                     self.base_ang_vel * self.obs_scales.ang_vel,  # dim 3
                     self._env_to_policy_all((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :-self.cfg.env.num_gripper_joints],  # dim 18
                     self._env_to_policy_all(self.dof_vel * self.obs_scales.dof_vel)[:, :-self.cfg.env.num_gripper_joints],  # dim 18
                     self._env_to_policy_all(self.action_history_buf[:, -1])[:, :12],  # dim 12
                     self.foot_contacts_from_sensor,  # dim 4
                     self.commands[:, :3] * self.commands_scale,  # dim 3
                     ee_goal_local_cart,  # dim 3 position
                     0 * self.curr_ee_goal_sphere]  # dim 3 orientation
        if self.cfg.goal_ee.sphere_center.mixed_height_reference:
            obs_terms.append(self.goal_height_follow_mask.float().unsqueeze(1))
        obs_buf = torch.cat(obs_terms, dim=-1)
        if self.cfg.env.observe_gait_commands:
            obs_buf = torch.cat((obs_buf,
                                      self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)
            
        if self.cfg.domain_rand.observe_priv:
            priv_buf = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[:, :12] - 1,
            ), dim=-1)
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )        

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        r, p, _ = euler_from_quat(self.base_quat) 
        z = self.root_states[:, 2]

        # r_threshold_buff = ((r > 0.2) & (self.curr_ee_goal_sphere[:, 2] >= 0)) | ((r < -0.2) & (self.curr_ee_goal_sphere[:, 2] <= 0))
        # p_threshold_buff = ((p > 0.2) & (self.curr_ee_goal_sphere[:, 1] >= 0)) | ((p < -0.2) & (self.curr_ee_goal_sphere[:, 1] <= 0))
        r_term = torch.abs(r) > 0.8
        p_term = torch.abs(p) > 0.8
        z_term = z < 0.1
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        if self.num_envs == 1 and self.is_main_process:
            if r_term:
                print("Terminated due to roll angle. Roll: ", r)
            elif p_term:
                print("Terminated due to pitch angle. Pitch: ", p)
            elif z_term:
                print("Terminated due to height. Height: ", z)
            elif termination_contact_buf:
                print("Terminated due to contact forces. Contact forces: ", self.contact_forces[:, self.termination_contact_indices, :])

        # arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        # arm_base = quat_apply(self.base_quat, arm_base_local) + self.root_states[:, :3]
        # curr_ee_pos_local = quat_rotate_inverse(self.root_states[:, 3:7], self.ee_pos - arm_base)
        # ik_fail = (self.curr_ee_goal_cart[:, -1:] - curr_ee_pos_local[:, -1:]).norm(dim=-1) > 0.3
        # self.reset_buf = termination_contact_buf | self.time_out_buf | r_term | p_term | z_term | ik_fail
        self.reset_buf = termination_contact_buf | self.time_out_buf | r_term | p_term | z_term

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain(self.cfg.terrain, verbose=self.is_main_process)
        self._create_trimesh()
        # self._create_ground_plane()
        self._create_envs()
        
    def reset_idx(self, env_ids, start=False):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        if self.cfg.env.teleop_mode and not self.teleop_initialize_targets_on_next_reset:
            self._cache_teleop_reset_state(env_ids)
        self._prepare_reset_initial_arm_pose(env_ids)
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_goal_height_reference(env_ids, is_init=start)

        if start:
            command_env_ids = env_ids
        else:
            command_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
        self._resample_commands(command_env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.goal_timer[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        for key in self.episode_metric_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_metric_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_metric_sums[key][env_ids] = 0.

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    # ------------ callbacks ------------

    def _parse_cfg(self, cfg):
        self.num_torques = self.cfg.env.num_torques
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.arm_reward_scales = class_to_dict(self.cfg.rewards.arm_scales)
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
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
        if not self.cfg.goal_ee.ranges.omnidirectional_pos_y:
            non_omni_pos_y_max = abs(resolve_schedule_value(
                self.cfg.commands.non_omni_pos_y_schedule,
                counter=schedule_counter,
                default_end_iter=schedule_total_iterations,
            ))
            self.goal_ee_ranges["pos_y"] = [-non_omni_pos_y_max, non_omni_pos_y_max]

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.action_delay = self.cfg.env.action_delay
        self.stop_update_goal = self.cfg.env.stop_update_goal
        self.record_video = self.cfg.env.record_video

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards 

        reward_contrainers = {"maniploco_rewards": ManipLoco_rewards}
        self.reward_container = reward_contrainers[self.cfg.rewards.reward_container_name](self)
        self.reward_scales = {k:v for k,v in self.reward_scales.items() if v is not None and v != 0}

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self.reward_container, name))

        self.arm_reward_scales = {k:v for k,v in self.arm_reward_scales.items() if v is not None and v != 0}

        # prepare list of functions
        self.arm_reward_functions = []
        self.arm_reward_names = []
        for name, scale in self.arm_reward_scales.items():
            if name=="termination":
                continue
            self.arm_reward_names.append(name)
            name = '_reward_' + name
            self.arm_reward_functions.append(getattr(self.reward_container, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}

        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                        for name in list(self.reward_scales.keys()) + list(self.arm_reward_scales.keys())}

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level  # start from 0
        if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        if self.is_main_process:
            print("Added ground plane with friction: {}, restitution: {}".format(plane_params.static_friction, plane_params.restitution))
        return
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        if self.is_main_process:
            print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        if self.is_main_process:
            print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_mesh_materials = True

        # Robot
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
        dof_props_asset['stiffness'][12:].fill(400.0)
        dof_props_asset['damping'][12:].fill(40.0)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_wo_gripper_names = self.dof_names[:-self.cfg.env.num_gripper_joints]
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        self.base_body_idx = self.body_names_to_idx[self.cfg.asset.base_name]
        self.policy_leg_joint_names = list(self.cfg.asset.policy_leg_joint_names)
        self.policy_leg_dof_indices_in_env = torch.tensor(
            [self.dof_names.index(name) for name in self.policy_leg_joint_names],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.env_leg_dof_indices_in_policy = torch.empty_like(self.policy_leg_dof_indices_in_env)
        self.env_leg_dof_indices_in_policy[self.policy_leg_dof_indices_in_env] = torch.arange(
            len(self.policy_leg_joint_names),
            device=self.device,
            dtype=torch.long,
        )
        # self.num_bodies = len(self.body_names)
        # self.num_dofs = len(self.dof_names)
        feet_names = list(self.cfg.asset.policy_foot_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            penalized_contact_names.extend(body_names)
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            body_names = [s for s in self.body_names if name in s]
            if len(body_names) == 0:
                raise Exception('No body found with name {}'.format(name))
            termination_contact_names.extend(body_names)

        self.sensor_indices = []
        for name in feet_names:
            foot_idx = self.body_names_to_idx[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)
        
        self.gripper_idx = self.body_names_to_idx[self.cfg.asset.gripper_name]

        # box
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, self.cfg.box.box_size, self.cfg.box.box_size, self.cfg.box.box_size, asset_options)

        if self.is_main_process:
            print('------------------------------------------------------')
            print('num_actions: {}'.format(self.num_actions))
            print('num_torques: {}'.format(self.num_torques))
            print('num_dofs: {}'.format(self.num_dofs))
            print('num_bodies: {}'.format(self.num_bodies))
            print('penalized_contact_names: {}'.format(penalized_contact_names))
            print('termination_contact_names: {}'.format(termination_contact_names))
            print('feet_names: {}'.format(feet_names))
            print(f"EE Gripper index: {self.gripper_idx}")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        box_start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.box_actor_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # widowGo1 
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-self.cfg.init_state.origin_perturb_range, self.cfg.init_state.origin_perturb_range, (2,1), device=self.device).squeeze(1)
            rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.init_state.rand_yaw_range*np.random.uniform(-1, 1))
            start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_dog_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot_dog", i, self.cfg.asset.self_collisions, 0)
            self.actor_handles.append(robot_dog_handle)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_dog_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_dog_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_dog_handle, body_props, recomputeInertia=True)
            
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)

            # box
            box_pos = pos.clone()
            box_pos[0] += 2
            box_pos[2] += self.cfg.box.box_env_origins_z
            box_start_pose.p = gymapi.Vec3(*box_pos)
            box_handle = self.gym.create_actor(env_handle, box_asset, box_start_pose, "box", i, self.cfg.asset.self_collisions, 0)
            self.box_actor_handles.append(box_handle)

            box_body_props = self.gym.get_actor_rigid_body_properties(env_handle, box_handle)
            box_body_props, _ = self._box_process_rigid_body_props(box_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, box_handle, box_body_props, recomputeInertia=True)

        self.robot_actor_indices = torch.arange(0, 2 * self.num_envs, 2, device=self.device)
        self.box_actor_indices = torch.arange(1, 2 * self.num_envs, 2, device=self.device)

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)

        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch.cat([
                    torch_rand_float(self.cfg.domain_rand.leg_motor_strength_range[0], self.cfg.domain_rand.leg_motor_strength_range[1], (self.num_envs, 12), device=self.device),
                    torch_rand_float(self.cfg.domain_rand.arm_motor_strength_range[0], self.cfg.domain_rand.arm_motor_strength_range[1], (self.num_envs, 6), device=self.device)
                ], dim=1)
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        hip_names = list(self.cfg.asset.hip_joint_names)
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        if self.is_main_process:
            print('penalized_contact_indices: {}'.format(self.penalized_contact_indices))
            print('termination_contact_indices: {}'.format(self.termination_contact_indices))
            print('feet_indices: {}'.format(self.feet_indices))

        if self.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1920
            camera_props.height = 1080
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([0, 1, 0.5])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
    
    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[self.base_body_idx].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
            gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
            props[self.gripper_idx].mass += gripper_rand_mass
        else:
            gripper_rand_mass = np.zeros(1)

        if self.cfg.domain_rand.randomize_base_com:
            rng_com_x = self.cfg.domain_rand.added_com_range_x
            rng_com_y = self.cfg.domain_rand.added_com_range_y
            rng_com_z = self.cfg.domain_rand.added_com_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[self.base_body_idx].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)

        mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass])
        return props, mass_params
    
    def _box_process_rigid_body_props(self, props, env_id):
        if self.cfg.box.randomize_base_mass:
            rng_mass = self.cfg.box.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros(1)
        
        return props, rand_mass

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 1000
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs, 1, 1)) 
        
        return props

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13) # 2 actors
        self.root_states = self._root_states[:, 0, :]
        self.box_root_state = self._root_states[:, 1, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.cfg.env.num_gripper_joints]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.cfg.env.num_gripper_joints]
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]
        urdf_mount_cfg = self.cfg.goal_ee.urdf_mount
        self.arm_base_offset = torch.tensor(
            urdf_mount_cfg.arm_base_offset,
            device=self.device,
            dtype=torch.float
        ).repeat(self.num_envs, 1)
        # Keep URDF-derived joint heights centralized in config.
        self.arm_waist_offset = self.arm_base_offset.clone()
        self.arm_waist_offset[:, 2] += urdf_mount_cfg.arm_waist_offset_z
        self.arm_shoulder_offset = self.arm_waist_offset.clone()
        self.arm_shoulder_offset[:, 2] += urdf_mount_cfg.arm_shoulder_offset_z
        # self.yaw_ema = euler_from_quat(self.base_quat)[2]
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.arm_mount_yaw_offset = float(urdf_mount_cfg.mount_yaw_offset)

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = self._contact_forces[:, :-1, :]
        self.box_contact_force = self._contact_forces[:, -1, :]

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_state = self._rigid_body_state[:, :-1, :]
        self.box_rigid_body_state = self._rigid_body_state[:, -1, :]

        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        # ee info
        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_vel = self.rigid_body_state[:, self.gripper_idx, 7:]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + self.cfg.env.num_gripper_joints):-self.cfg.env.num_gripper_joints]

        # box info & target_ee info
        self.box_pos = self.box_root_state[:, 0:3]
        self.grasp_offset = self.cfg.arm.grasp_offset
        self.init_target_ee_base = torch.tensor(self.cfg.arm.init_target_ee_base, device=self.device).unsqueeze(0)

        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2])
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.teleop_raw_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.teleop_raw_ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        self.reset_init_ee_sphere = self.init_start_ee_sphere.repeat(self.num_envs, 1)
        self.reset_init_arm_waist_yaw = torch.zeros(self.num_envs, device=self.device)
        self.omnidirectional_pos_y = bool(self.cfg.goal_ee.ranges.omnidirectional_pos_y)
        self.omnidirectional_rear_transition_pos_y_abs = float(self.cfg.goal_ee.ranges.omnidirectional_rear_transition_pos_y_abs)
        if not (0.0 <= self.omnidirectional_rear_transition_pos_y_abs < np.pi):
            raise ValueError("goal_ee.ranges.omnidirectional_rear_transition_pos_y_abs must be in [0, pi)")
        self.omnidirectional_pos_l = torch.tensor(self.cfg.goal_ee.ranges.omnidirectional_pos_l, device=self.device, dtype=torch.float)
        self.omnidirectional_rear_pos_l = torch.tensor(self.cfg.goal_ee.ranges.omnidirectional_rear_pos_l, device=self.device, dtype=torch.float)
        self.omnidirectional_rear_pos_p = torch.tensor(self.cfg.goal_ee.ranges.omnidirectional_rear_pos_p, device=self.device, dtype=torch.float)

        #noise
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.add_noise = self.cfg.noise.add_noise

        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.ee_goal_center_offset = torch.tensor([self.cfg.goal_ee.sphere_center.x_offset, 
                                                   self.cfg.goal_ee.sphere_center.y_offset, 
                                                   self.cfg.goal_ee.sphere_center.z_invariant_offset], 
                                                   device=self.device).repeat(self.num_envs, 1)
        self.goal_height_follow_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.goal_height_follow_override = None
        
        self.curr_ee_goal_cart_world = self.transform_goal_local_to_world(self.get_goal_center_offset_local() + self.curr_ee_goal_cart)

        if self.is_main_process:
            print('------------------------------------------------------')
            print(f'root_states shape: {self.root_states.shape}')
            print(f'dof_state shape: {self.dof_state.shape}')
            print(f'force_sensor_tensor shape: {self.force_sensor_tensor.shape}')
            print(f'contact_forces shape: {self.contact_forces.shape}')
            print(f'rigid_body_state shape: {self.rigid_body_state.shape}')
            print(f'jacobian_whole shape: {self.jacobian_whole.shape}')
            print(f'box_root_state shape: {self.box_root_state.shape}')
            print(f'box_contact_force shape: {self.box_contact_force.shape}')
            print(f'box_rigid_body_state shape: {self.box_rigid_body_state.shape}')
            print('------------------------------------------------------')
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.extras["episode"] = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.teleop_raw_commands = torch.zeros_like(self.commands)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        # Refer to <walk these ways>, only useful when `self.cfg.env.observe_gait_commands` is True
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
        
        # self.target_ee = torch.zeros(self.num_envs, self.cfg.target_ee.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # ee x, ee y, ee z
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)
        self.gripper_pos_targets = torch.zeros(self.num_envs, self.cfg.env.num_gripper_joints, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        
        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            if name in self.cfg.control.stiffness:
                self.p_gains[i] = self.cfg.control.stiffness[name]
                self.d_gains[i] = self.cfg.control.damping[name]
                found = True
            else:
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                        self.d_gains[i] = self.cfg.control.damping[dof_name]
                        found = True
                        break
            if not found:
                for dof_name in self.cfg.control.damping.keys():
                    if dof_name in name and dof_name in self.cfg.control.stiffness:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                        self.d_gains[i] = self.cfg.control.damping[dof_name]
                        found = True
                        break
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    raise Exception(f"PD gain of joint {name} were not defined, setting them to zero")
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-self.cfg.env.num_gripper_joints]
        self.gripper_pos_targets[:] = self.default_dof_pos[-self.cfg.env.num_gripper_joints:]
        self.arm_dof_start_idx = self.num_dofs - (6 + self.cfg.env.num_gripper_joints)
        self.arm_dof_end_idx = self.num_dofs - self.cfg.env.num_gripper_joints
        self.teleop_arm_control_mode = "ee"
        self.teleop_arm_joint_step = 0.05
        self.teleop_arm_joint_pos_targets = self.dof_pos[:, self.arm_dof_start_idx:self.arm_dof_end_idx].clone()
        self.teleop_hold_actual_ee_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.teleop_initialize_targets_on_next_reset = True
        self.teleop_restore_arm_gripper_state_on_reset = bool(
            getattr(self.cfg.env, "teleop_restore_arm_gripper_state_on_reset", False)
        )
        self.teleop_saved_arm_dof_pos = self.teleop_arm_joint_pos_targets.clone()
        self.teleop_saved_arm_joint_targets = self.teleop_arm_joint_pos_targets.clone()
        self.teleop_saved_gripper_dof_pos = self.gripper_pos_targets.clone()
        self.teleop_saved_gripper_pos_targets = self.gripper_pos_targets.clone()
        self.teleop_repeatable_actions = {
            "forward",
            "reverse",
            "turn_left",
            "turn_right",
            "increase_eef_goal_l",
            "decrease_eef_goal_l",
            "increase_eef_goal_p",
            "decrease_eef_goal_p",
            "increase_eef_goal_y",
            "decrease_eef_goal_y",
            "increase_eef_goal_dr",
            "decrease_eef_goal_dr",
            "increse_eef_goal_dp",
            "decrease_eef_goal_dp",
            "increase_eef_goal_dy",
            "decrease_eef_goal_dy",
            "open_gripper",
            "close_gripper",
        }
        self._register_repeatable_actions(*self.teleop_repeatable_actions)
        self.arm_waist_idx = self.dof_names.index(self.cfg.asset.arm_waist_name)
        self.omnidirectional_init_pos_y_limits = torch.tensor(
            self.cfg.goal_ee.ranges.omnidirectional_init_pos_y,
            device=self.device,
            dtype=torch.float,
        )
        self.arm_waist_reset_limits = torch.stack((
            torch.maximum(self.dof_pos_limits[self.arm_waist_idx, 0], self.omnidirectional_init_pos_y_limits[0]),
            torch.minimum(self.dof_pos_limits[self.arm_waist_idx, 1], self.omnidirectional_init_pos_y_limits[1]),
        ))
        
        self.global_steps = 0

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-self.cfg.init_state.origin_perturb_range, self.cfg.init_state.origin_perturb_range, (len(env_ids), 2), device=self.device) # xy position within 1m of the center

        self.box_root_state[env_ids, 0] = self.env_origins[env_ids, 0] + 2
        self.box_root_state[env_ids, 1] = self.env_origins[env_ids, 1]
        self.box_root_state[env_ids, 2] = self.env_origins[env_ids, 2] + self.cfg.box.box_env_origins_z
        # base orientation
        rand_yaw = self.cfg.init_state.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :]  
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-self.cfg.init_state.init_vel_perturb_range, self.cfg.init_state.init_vel_perturb_range, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 7:9] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            self.root_states[:, 7:9] * 2.5,
            self.root_states[:, 7:9]
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        arm_waist_pos = self.reset_init_arm_waist_yaw[env_ids] + torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze(-1)
        self.dof_pos[env_ids, self.arm_waist_idx] = torch.clamp(
            arm_waist_pos, min=self.dof_pos_limits[self.arm_waist_idx, 0], max=self.dof_pos_limits[self.arm_waist_idx, 1])
        self.dof_vel[env_ids] = 0.
        self.gripper_pos_targets[env_ids] = self.default_dof_pos[-self.cfg.env.num_gripper_joints:]
        if self.cfg.env.teleop_mode and not self.teleop_initialize_targets_on_next_reset:
            arm_slice = slice(self.arm_dof_start_idx, self.arm_dof_end_idx)
            gripper_slice = slice(self.num_dofs - self.cfg.env.num_gripper_joints, self.num_dofs)
            if self.teleop_restore_arm_gripper_state_on_reset:
                self.dof_pos[env_ids, arm_slice] = self.teleop_saved_arm_dof_pos[env_ids]
                self.dof_pos[env_ids, gripper_slice] = self.teleop_saved_gripper_dof_pos[env_ids]
            self.teleop_arm_joint_pos_targets[env_ids] = self.teleop_saved_arm_joint_targets[env_ids]
            self.gripper_pos_targets[env_ids] = self.teleop_saved_gripper_pos_targets[env_ids]
        else:
            self.teleop_arm_joint_pos_targets[env_ids] = self.dof_pos[env_ids, self.arm_dof_start_idx:self.arm_dof_end_idx]
            self.teleop_hold_actual_ee_target[env_ids] = False

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _cache_teleop_reset_state(self, env_ids):
        arm_slice = slice(self.arm_dof_start_idx, self.arm_dof_end_idx)
        gripper_slice = slice(self.num_dofs - self.cfg.env.num_gripper_joints, self.num_dofs)
        self.teleop_saved_arm_dof_pos[env_ids] = self.dof_pos[env_ids, arm_slice]
        self.teleop_saved_arm_joint_targets[env_ids] = self.teleop_arm_joint_pos_targets[env_ids]
        self.teleop_saved_gripper_dof_pos[env_ids] = self.dof_pos[env_ids, gripper_slice]
        self.teleop_saved_gripper_pos_targets[env_ids] = self.gripper_pos_targets[env_ids]

    def _prepare_reset_initial_arm_pose(self, env_ids):
        self.reset_init_ee_sphere[env_ids] = self.init_start_ee_sphere.repeat(len(env_ids), 1)
        self.reset_init_ee_sphere[env_ids, 2] = wrap_to_pi(self.reset_init_ee_sphere[env_ids, 2] + self.arm_mount_yaw_offset)
        if self.cfg.env.teleop_mode:
            return
        if self.omnidirectional_pos_y:
            yaw_min = float(self.arm_waist_reset_limits[0].item())
            yaw_max = float(self.arm_waist_reset_limits[1].item())
        else:
            yaw_min = self.goal_ee_ranges["pos_y"][0]
            yaw_max = self.goal_ee_ranges["pos_y"][1]
        sampled_yaw = torch_rand_float(yaw_min, yaw_max, (len(env_ids), 1), device=self.device).squeeze(1)
        self.reset_init_arm_waist_yaw[env_ids] = sampled_yaw
        self.reset_init_ee_sphere[env_ids, 2] = wrap_to_pi(sampled_yaw + self.arm_mount_yaw_offset)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if self.cfg.env.teleop_mode:
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

    def _update_effective_teleop_inputs(self):
        if not self.cfg.env.teleop_mode:
            return

        self.commands[:] = self.teleop_raw_commands
        if self.cfg.env.teleop_input_regularization:
            self.commands[:, 0] = torch.clip(
                self.commands[:, 0],
                -self.cfg.env.teleop_lin_vel_x_limit,
                self.cfg.env.teleop_lin_vel_x_limit,
            )
            self.commands[:, 2] = torch.clip(
                self.commands[:, 2],
                -self.cfg.env.teleop_ang_vel_yaw_limit,
                self.cfg.env.teleop_ang_vel_yaw_limit,
            )
            active_mask = torch.logical_or(
                torch.abs(self.commands[:, 0]) > self.cfg.env.teleop_zero_lin_vel_x_clip,
                torch.abs(self.commands[:, 2]) > self.cfg.env.teleop_zero_ang_vel_yaw_clip
            ).unsqueeze(1)
            self.commands[:] *= active_mask

        if self.teleop_arm_control_mode == "joint":
            self._sync_teleop_ee_goal_to_current_pose()
            return

        hold_mask = self.teleop_hold_actual_ee_target.unsqueeze(1)
        if self.cfg.env.teleop_input_regularization:
            clipped_ee_goal_cart = self.teleop_raw_ee_goal_cart.clone()
            clipped_ee_goal_cart[:, 0] = torch.clip(
                clipped_ee_goal_cart[:, 0],
                self.cfg.env.teleop_ee_goal_x_limit[0],
                self.cfg.env.teleop_ee_goal_x_limit[1],
            )
            clipped_ee_goal_cart[:, 1] = torch.clip(
                clipped_ee_goal_cart[:, 1],
                self.cfg.env.teleop_ee_goal_y_limit[0],
                self.cfg.env.teleop_ee_goal_y_limit[1],
            )
            clipped_ee_goal_cart[:, 2] = torch.clip(
                clipped_ee_goal_cart[:, 2],
                self.cfg.env.teleop_ee_goal_z_limit[0],
                self.cfg.env.teleop_ee_goal_z_limit[1],
            )
            clipped_ee_goal_delta_rpy = torch.clip(
                self.teleop_raw_ee_goal_orn_delta_rpy,
                min=torch.tensor(self.cfg.goal_ee.ranges.delta_orn_r[0:1] + self.cfg.goal_ee.ranges.delta_orn_p[0:1] + self.cfg.goal_ee.ranges.delta_orn_y[0:1], device=self.device),
                max=torch.tensor(self.cfg.goal_ee.ranges.delta_orn_r[1:2] + self.cfg.goal_ee.ranges.delta_orn_p[1:2] + self.cfg.goal_ee.ranges.delta_orn_y[1:2], device=self.device),
            )
            self.curr_ee_goal_cart[:] = torch.where(hold_mask, self.teleop_raw_ee_goal_cart, clipped_ee_goal_cart)
            self.ee_goal_orn_delta_rpy[:] = torch.where(hold_mask, self.teleop_raw_ee_goal_orn_delta_rpy, clipped_ee_goal_delta_rpy)
        else:
            self.curr_ee_goal_cart[:] = self.teleop_raw_ee_goal_cart
            self.ee_goal_orn_delta_rpy[:] = self.teleop_raw_ee_goal_orn_delta_rpy

        self.curr_ee_goal_sphere[:] = cart2sphere(self.curr_ee_goal_cart)

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

            def _compute_smoothing_multiplier(idxs):
                phase = torch.remainder(idxs, 1.0)
                return (
                    smoothing_cdf_start(phase) * (1 - smoothing_cdf_start(phase - 0.5))
                    + smoothing_cdf_start(phase - 1) * (1 - smoothing_cdf_start(phase - 1.5))
                )

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multipliers = {
                foot_name: _compute_smoothing_multiplier(shaped_foot_indices[foot_name])
                for foot_name in policy_foot_names
            }

            for i, foot_name in enumerate(policy_foot_names):
                self.desired_contact_states[:, i] = smoothing_multipliers[foot_name]

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
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        command_env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(command_env_ids)
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        if self.num_actions != 18:
            raise NotImplementedError("Noise scale is only implemented for action space of 12")

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        idx = 0

        # Body orientation (dim 2)
        noise_vec[idx:idx+2] = 0  # Assuming no noise for body orientation
        idx += 2

        # Angular velocity (dim 3)   
        noise_vec[idx:idx+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3

        # DOF positions (dim 12) b1
        noise_vec[idx:idx+12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += 12

        # DOF positions (dim 6) z1
        noise_vec[idx:idx+6] = 0
        idx += 6

        # DOF velocities (dim 12)
        noise_vec[idx:idx+12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += 12

        # DOF velocities (dim 6)
        noise_vec[idx:idx+6] = 0
        idx += 6

        # Action history (dim 12)
        noise_vec[idx:idx+12] = 0  # Assuming no noise for action history
        idx += 12

        # Foot contacts (dim 4)
        noise_vec[idx:idx+4] = 0  # Assuming no noise for foot contacts
        idx += 4

        # Commands (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for commands
        idx += 3

        # End-effector goal position (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for end-effector goal position
        idx += 3

        # End-effector goal orientation (dim 3)
        noise_vec[idx:idx+3] = 0  # Assuming no noise for end-effector goal orientation
        idx += 3

        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec
    
    def _env_to_policy_all(self, vec):
        if not self.cfg.env.reorder_dofs:
            return vec
        return torch.hstack((vec[:, self.env_leg_dof_indices_in_policy], vec[:, 12:]))

    def _policy_to_env_all(self, vec):
        if not self.cfg.env.reorder_dofs:
            return vec
        return torch.hstack((vec[:, self.policy_leg_dof_indices_in_env], vec[:, 12:]))

    def _env_to_policy_dog(self, vec):
        if not self.cfg.env.reorder_dofs:
            return vec
        return vec[:, self.env_leg_dof_indices_in_policy]
    
    def _get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = torch.stack([r, p, y], dim=-1)

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles

    def get_goal_reference_quat(self):
        """Returns the goal-reference orientation in world coordinates.

        The returned quaternion maps vectors from the goal local frame to world.
        z-invariant uses yaw-only; trunk-follow uses the full trunk 6DoF orientation.
        """
        if not self.cfg.goal_ee.sphere_center.mixed_height_reference:
            return self.base_yaw_quat
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)

    def get_goal_reference_origin(self):
        """Returns the goal-reference origin in world coordinates.

        z-invariant uses `(root_x, root_y, 0)`.
        trunk-follow uses the full root position `root_states[:, :3]`.
        """
        invariant_origin = torch.cat(
            [self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)],
            dim=1,
        )
        if not self.cfg.goal_ee.sphere_center.mixed_height_reference:
            return invariant_origin
        return torch.where(self.goal_height_follow_mask.unsqueeze(1), self.root_states[:, :3], invariant_origin)

    def get_goal_center_offset_local(self):
        """Returns the target-center offset in the goal local frame.

        In z-invariant mode this is the sphere-center config offset.
        In trunk-follow mode this switches to the configured URDF anchor: arm base, waist, or shoulder.
        """
        if not self.cfg.goal_ee.sphere_center.mixed_height_reference:
            return self.ee_goal_center_offset
        trunk_follow_anchor = self.cfg.goal_ee.sphere_center.trunk_follow_anchor
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

    def _get_arm_base_world_pos(self):
        """Returns the arm-base origin in world coordinates.

        This is the URDF `link00` mounting point attached to the robot base/trunk.
        """
        arm_base_quat = self.base_yaw_quat
        if self.cfg.goal_ee.sphere_center.mixed_height_reference:
            arm_base_quat = torch.where(self.goal_height_follow_mask.unsqueeze(1), self.base_quat, self.base_yaw_quat)
        return self.base_pos + quat_apply(arm_base_quat, self.arm_base_offset)

    def _draw_collision_bbox(self, env_ids=None):
        """Draws the red forbidden box in world coordinates.

        The box corners are defined by `collision_lower_limits/upper_limits` in the goal local frame,
        then transformed into world with the active goal reference frame.
        """
        if env_ids is None:
            env_ids = range(self.num_envs)

        center = self.get_goal_center_offset_local()
        bbox0 = center + self.collision_upper_limits
        bbox1 = center + self.collision_lower_limits
        bboxes = torch.stack([bbox0, bbox1], dim=1)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        goal_ref_origin = self.get_goal_reference_origin()
        goal_ref_quat = self.get_goal_reference_quat()

        for i in env_ids:
            bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))
            quat = goal_ref_quat[i]
            r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            pose0 = gymapi.Transform(gymapi.Vec3(goal_ref_origin[i, 0], goal_ref_origin[i, 1], goal_ref_origin[i, 2]), r=r)
            gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0) 

    def _draw_ee_goal_curr(self, env_ids=None):
        """Draws the current goal markers in world coordinates.

        Yellow = current EE target point in world.
        Cyan = goal-frame center in world.
        White = root/base pose in world.
        Blue = measured gripper position in world.
        """
        if env_ids is None:
            env_ids = range(self.num_envs)

        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        sphere_geom_root = gymutil.WireframeSphereGeometry(0.06, 16, 16, None, color=(1, 1, 1))

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))
        upper_arm_pose = self.get_ee_goal_spherical_center()

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))
        ee_pose = self.rigid_body_state[:, self.gripper_idx, :3]

        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)
        if 0 in env_ids:
            gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)

        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in env_ids:
            sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

            root_pose = gymapi.Transform(
                gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], self.root_states[i, 2]),
                r=gymapi.Quat(self.base_quat[i, 0], self.base_quat[i, 1], self.base_quat[i, 2], self.base_quat[i, 3])
            )
            gymutil.draw_lines(sphere_geom_root, self.gym, self.viewer, self.envs[i], root_pose)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], root_pose)

            pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), 
                                    r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_ee_goal_traj(self, env_ids=None):
        """Draws the sampled EE goal trajectory in world coordinates."""
        if env_ids is None:
            env_ids = range(self.num_envs)

        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))
        sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[:, None, None]
        ee_target_all_sphere = self._interpolate_goal_sphere(
            self.ee_start_sphere[None, ...],
            self.ee_goal_sphere[None, ...],
            t,
        )
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        goal_ref_quat = self.get_goal_reference_quat()
        goal_ref_center = self.get_ee_goal_spherical_center()
        for i in range(10):
            ee_target_cart = sphere2cart(ee_target_all_sphere[i])
            ee_target_all_cart_world[i] = quat_apply(goal_ref_quat, ee_target_cart)
        ee_target_all_cart_world += goal_ref_center[None, :, :]
        for i in env_ids:
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[j, i, 0], ee_target_all_cart_world[j, i, 1], ee_target_all_cart_world[j, i, 2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _control_ik(self, dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))#.view(self.num_envs, 6)
        return u.squeeze(-1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.motor_strength * self.action_scale

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        default_torques[:, -6:] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _get_omnidirectional_goal_sampling_bounds(self, goal_yaw):
        rear_weight = self._get_omnidirectional_rear_weight(goal_yaw)

        pos_l_min_front = torch.full_like(goal_yaw, self.omnidirectional_pos_l[0].item())
        pos_l_min_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_l[0].item())
        pos_l_min = torch.lerp(pos_l_min_front, pos_l_min_back, rear_weight)
        pos_l_max_front = torch.full_like(goal_yaw, self.omnidirectional_pos_l[1].item())
        pos_l_max_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_l[1].item())
        pos_l_max = torch.lerp(pos_l_max_front, pos_l_max_back, rear_weight)

        pos_p_min_front = torch.full_like(goal_yaw, float(self.goal_ee_ranges["pos_p"][0]))
        pos_p_max_front = torch.full_like(goal_yaw, float(self.goal_ee_ranges["pos_p"][1]))
        pos_p_min_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_p[0].item())
        pos_p_max_back = torch.full_like(goal_yaw, self.omnidirectional_rear_pos_p[1].item())
        pos_p_min = torch.lerp(pos_p_min_front, pos_p_min_back, rear_weight)
        pos_p_max = torch.lerp(pos_p_max_front, pos_p_max_back, rear_weight)

        return pos_l_min, pos_l_max, pos_p_min, pos_p_max

    def _get_omnidirectional_rear_weight(self, goal_yaw):
        abs_yaw = torch.abs(wrap_to_pi(goal_yaw))
        denom = np.pi - self.omnidirectional_rear_transition_pos_y_abs
        transition = (abs_yaw - self.omnidirectional_rear_transition_pos_y_abs) / denom
        transition = torch.clamp(transition, 0.0, 1.0)
        return transition * transition * (3.0 - 2.0 * transition)

    def _interpolate_goal_sphere(self, start_sphere, goal_sphere, t):
        interp_sphere = torch.lerp(start_sphere, goal_sphere, t)
        yaw_t = t[..., 0] if t.ndim > 0 and t.shape[-1] == 1 else t
        start_yaw_arm_front = wrap_to_pi(start_sphere[..., 2] - self.arm_mount_yaw_offset)
        goal_yaw_arm_front = wrap_to_pi(goal_sphere[..., 2] - self.arm_mount_yaw_offset)
        interp_sphere[..., 2] = wrap_to_pi(
            torch.lerp(start_yaw_arm_front, goal_yaw_arm_front, yaw_t) + self.arm_mount_yaw_offset
        )
        return interp_sphere

    def _resample_goal_height_reference(self, env_ids, is_init=False):
        if not self.cfg.goal_ee.sphere_center.mixed_height_reference or len(env_ids) == 0:
            return

        if self.cfg.env.teleop_mode and is_init:
            if self.goal_height_follow_override is None:
                self.goal_height_follow_mask[env_ids] = False
            else:
                self.goal_height_follow_mask[env_ids] = self.goal_height_follow_override
            return
        elif self.cfg.env.teleop_mode:
            return

        trunk_follow_ratio = self.cfg.goal_ee.sphere_center.trunk_follow_ratio
        self.goal_height_follow_mask[env_ids] = torch.rand(len(env_ids), device=self.device) < trunk_follow_ratio

    def _resample_ee_goal(self, env_ids, is_init=False):
        if self.cfg.env.teleop_mode and is_init:
            if self.teleop_initialize_targets_on_next_reset:
                self._reset_teleop_ee_goal_to_default()
                self.teleop_initialize_targets_on_next_reset = False
            return
        elif self.cfg.env.teleop_mode:
            return

        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            
            if is_init:
                self.ee_goal_orn_delta_rpy[env_ids, :] = 0
                self.ee_start_sphere[env_ids] = self.reset_init_ee_sphere[env_ids].clone()
            else:
                self._resample_ee_goal_orn_once(env_ids)
                self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self._collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
            self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0

    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), device=self.device)
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)

    def _resample_ee_goal_sphere_once(self, env_ids):
        sampled_yaw = torch_rand_float(self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.omnidirectional_pos_y:
            start_yaw = wrap_to_pi(self.ee_start_sphere[env_ids, 2] - self.arm_mount_yaw_offset)
            goal_yaw = torch.clamp(sampled_yaw + start_yaw, min=-np.pi, max=np.pi)
            self.ee_goal_sphere[env_ids, 2] = wrap_to_pi(goal_yaw + self.arm_mount_yaw_offset)
            pos_l_min, pos_l_max, pos_p_min, pos_p_max = self._get_omnidirectional_goal_sampling_bounds(goal_yaw)
            self.ee_goal_sphere[env_ids, 0] = pos_l_min + (pos_l_max - pos_l_min) * torch.rand(len(env_ids), device=self.device)
            self.ee_goal_sphere[env_ids, 1] = pos_p_min + (pos_p_max - pos_p_min) * torch.rand(len(env_ids), device=self.device)
            return

        self.ee_goal_sphere[env_ids, 2] = wrap_to_pi(sampled_yaw + self.arm_mount_yaw_offset)
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _collision_check(self, env_ids):
        """Rejects EE trajectories that enter the forbidden box in goal local coordinates.

        The interpolated path is sampled in the goal local frame, then compared against
        `collision_lower_limits/upper_limits` and `underground_limit`.
        """
        ee_target_all_sphere = self._interpolate_goal_sphere(
            self.ee_start_sphere[env_ids][None, ...],
            self.ee_goal_sphere[env_ids][None, ...],
            self.collision_check_t.squeeze(0).squeeze(0)[:, None, None],
        )
        ee_target_cart = sphere2cart(ee_target_all_sphere.reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def _reset_teleop_ee_goal_to_default(self):
        self.curr_ee_goal_sphere[:] = self.reset_init_ee_sphere[:]
        self.teleop_raw_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        self.teleop_raw_ee_goal_orn_delta_rpy[:] = 0
        self.teleop_hold_actual_ee_target[:] = False
        self._update_effective_teleop_inputs()

    def _sync_teleop_arm_joint_targets_to_current_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        arm_limits = self.dof_pos_limits[self.arm_dof_start_idx:self.arm_dof_end_idx]
        current_arm_pos = self.dof_pos[env_ids, self.arm_dof_start_idx:self.arm_dof_end_idx]
        self.teleop_arm_joint_pos_targets[env_ids] = torch.clip(
            current_arm_pos,
            arm_limits[:, 0],
            arm_limits[:, 1],
        )

    def _sync_teleop_ee_goal_to_current_pose(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        goal_ref_origin = self.get_goal_reference_origin()[env_ids]
        goal_ref_quat = self.get_goal_reference_quat()[env_ids]
        goal_center_offset = self.get_goal_center_offset_local()[env_ids]
        ee_orn_normalized = self.ee_orn[env_ids] / torch.norm(self.ee_orn[env_ids], dim=-1, keepdim=True)

        ee_goal_local_with_center = quat_rotate_inverse(goal_ref_quat, self.ee_pos[env_ids] - goal_ref_origin)
        self.teleop_raw_ee_goal_cart[env_ids] = ee_goal_local_with_center - goal_center_offset
        self.curr_ee_goal_cart[env_ids] = self.teleop_raw_ee_goal_cart[env_ids]
        self.curr_ee_goal_sphere[env_ids] = cart2sphere(self.curr_ee_goal_cart[env_ids])

        local_ee_orn = quat_mul(quat_conjugate(goal_ref_quat), ee_orn_normalized)
        local_ee_orn_rpy = torch.stack(euler_from_quat(local_ee_orn), dim=-1)
        default_pitch = -self.curr_ee_goal_sphere[env_ids, 1] + self.cfg.goal_ee.arm_induced_pitch
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 0] = wrap_to_pi(local_ee_orn_rpy[:, 0] - np.pi / 2)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 1] = wrap_to_pi(local_ee_orn_rpy[:, 1] - default_pitch)
        self.teleop_raw_ee_goal_orn_delta_rpy[env_ids, 2] = wrap_to_pi(local_ee_orn_rpy[:, 2] - self.curr_ee_goal_sphere[env_ids, 2])
        self.ee_goal_orn_delta_rpy[env_ids] = self.teleop_raw_ee_goal_orn_delta_rpy[env_ids]
        self.teleop_hold_actual_ee_target[env_ids] = True

        self.curr_ee_goal_cart_world[env_ids] = self.ee_pos[env_ids]
        self.ee_goal_orn_quat[env_ids] = ee_orn_normalized
        self.ee_goal_orn_euler[env_ids] = torch.stack(euler_from_quat(self.ee_goal_orn_quat[env_ids]), dim=-1)

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

    def _toggle_teleop_arm_control_mode(self):
        next_mode = "joint" if self.teleop_arm_control_mode == "ee" else "ee"
        self._set_teleop_arm_control_mode(next_mode)

    def _update_curr_ee_goal(self):
        """Updates the end-effector target.

        `curr_ee_goal_cart` stays in the goal local frame.
        `curr_ee_goal_cart_world` is the yellow target point in world coordinates.
        `ee_goal_orn_quat` is the target orientation in world coordinates.
        """
        if not self.cfg.env.teleop_mode:
            t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
            self.curr_ee_goal_sphere[:] = self._interpolate_goal_sphere(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        else:
            self._update_effective_teleop_inputs()

        # TODO: for the teleop mode, we need to directly update self.curr_ee_goal_cart using VR controller.
        if not self.cfg.env.teleop_mode:
            self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        goal_ref_quat = self.get_goal_reference_quat()
        self.curr_ee_goal_cart_world = self.transform_goal_local_to_world(self.get_goal_center_offset_local() + self.curr_ee_goal_cart)
        
        # TODO: for the teleop mode, we need to directly update self.ee_goal_orn_quat using VR controller.
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        local_goal_orn = quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + self.curr_ee_goal_sphere[:, 2],
        )
        self.ee_goal_orn_quat = quat_mul(goal_ref_quat, local_goal_orn)
        self.ee_goal_orn_euler = torch.stack(euler_from_quat(self.ee_goal_orn_quat), dim=-1)
        
        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        
        if len(resample_id) > 0 and self.stop_update_goal:
            # set these env commands as 0
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 2] = 0
            if self.cfg.env.teleop_mode:
                self.teleop_raw_commands[resample_id, 0] = 0
                self.teleop_raw_commands[resample_id, 2] = 0

        self._resample_ee_goal(resample_id)

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
    

   # -------------- IssacGym render and viewer functions ----------------

    def render_record(self, mode="rgb_array"):
        if self.global_steps % 2 == 0:
            self.gym.step_graphics(self.sim)
            imgs = []
            for i in range(self.num_envs):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([0, 2, 1])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                self.gym.clear_lines(self.viewer)
                self._draw_ee_goal_curr(env_ids=[i])
                self._draw_ee_goal_traj(env_ids=[i])
                self._draw_collision_bbox(env_ids=[i])
                self.gym.render_all_camera_sensors(self.sim)
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            self.gym.clear_lines(self.viewer)
            return imgs
        return None

    def subscribe_viewer_keyboard_events(self):
        super().subscribe_viewer_keyboard_events()

        if self.cfg.env.teleop_mode:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_linear")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "increase_eef_goal_l")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "decrease_eef_goal_l")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "increase_eef_goal_p")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "decrease_eef_goal_p")  
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "increase_eef_goal_y")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "decrease_eef_goal_y")

            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "increase_eef_goal_dr")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "decrease_eef_goal_dr")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "increse_eef_goal_dp")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "decrease_eef_goal_dp")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "increase_eef_goal_dy")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "decrease_eef_goal_dy")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "open_gripper")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "close_gripper")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "reset_eef_goal_pose")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "toggle_arm_control_mode")
            if self.cfg.goal_ee.sphere_center.mixed_height_reference:
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "set_height_reference_invariant")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "set_height_reference_follow")

    def _apply_teleop_action(self, action):
        if action == "stop_linear":
            self.teleop_raw_commands[:, 0] = 0
        elif action == "forward":
            self.teleop_raw_commands[:, 0] += 0.05
        elif action == "reverse":
            self.teleop_raw_commands[:, 0] -= 0.05

        if action == "stop_angular":
            self.teleop_raw_commands[:, 2] = 0
        elif action == "turn_left":
            self.teleop_raw_commands[:, 2] += 0.05
        elif action == "turn_right":
            self.teleop_raw_commands[:, 2] -= 0.05

        if action == "toggle_arm_control_mode":
            self._toggle_teleop_arm_control_mode()

        self._update_effective_teleop_inputs()

        if self.teleop_arm_control_mode == "joint":
            joint_delta = self.teleop_arm_joint_step
            if action == "increase_eef_goal_l":
                self.teleop_arm_joint_pos_targets[:, 0] += joint_delta
            elif action == "decrease_eef_goal_l":
                self.teleop_arm_joint_pos_targets[:, 0] -= joint_delta

            if action == "increase_eef_goal_p":
                self.teleop_arm_joint_pos_targets[:, 1] += joint_delta
            elif action == "decrease_eef_goal_p":
                self.teleop_arm_joint_pos_targets[:, 1] -= joint_delta

            if action == "increase_eef_goal_y":
                self.teleop_arm_joint_pos_targets[:, 2] += joint_delta
            elif action == "decrease_eef_goal_y":
                self.teleop_arm_joint_pos_targets[:, 2] -= joint_delta

            if action == "increase_eef_goal_dr":
                self.teleop_arm_joint_pos_targets[:, 3] += joint_delta
            elif action == "decrease_eef_goal_dr":
                self.teleop_arm_joint_pos_targets[:, 3] -= joint_delta

            if action == "increse_eef_goal_dp":
                self.teleop_arm_joint_pos_targets[:, 4] += joint_delta
            elif action == "decrease_eef_goal_dp":
                self.teleop_arm_joint_pos_targets[:, 4] -= joint_delta

            if action == "increase_eef_goal_dy":
                self.teleop_arm_joint_pos_targets[:, 5] += joint_delta
            elif action == "decrease_eef_goal_dy":
                self.teleop_arm_joint_pos_targets[:, 5] -= joint_delta

            if action == "reset_eef_goal_pose":
                self.teleop_arm_joint_pos_targets[:] = self.default_dof_pos[self.arm_dof_start_idx:self.arm_dof_end_idx]
                print("[teleop] reset arm joint targets to default")

            arm_limits = self.dof_pos_limits[self.arm_dof_start_idx:self.arm_dof_end_idx]
            self.teleop_arm_joint_pos_targets[:] = torch.clip(
                self.teleop_arm_joint_pos_targets,
                arm_limits[:, 0],
                arm_limits[:, 1],
            )
        else:
            if action in {
                "increase_eef_goal_l",
                "decrease_eef_goal_l",
                "increase_eef_goal_p",
                "decrease_eef_goal_p",
                "increase_eef_goal_y",
                "decrease_eef_goal_y",
                "increase_eef_goal_dr",
                "decrease_eef_goal_dr",
                "increse_eef_goal_dp",
                "decrease_eef_goal_dp",
                "increase_eef_goal_dy",
                "decrease_eef_goal_dy",
            }:
                self.teleop_hold_actual_ee_target[:] = False

            if action == "increase_eef_goal_l":
                self.teleop_raw_ee_goal_cart[:, 0] += 0.05
            elif action == "decrease_eef_goal_l":
                self.teleop_raw_ee_goal_cart[:, 0] -= 0.05

            if action == "increase_eef_goal_p":
                self.teleop_raw_ee_goal_cart[:, 1] += 0.05
            elif action == "decrease_eef_goal_p":
                self.teleop_raw_ee_goal_cart[:, 1] -= 0.05

            if action == "increase_eef_goal_y":
                self.teleop_raw_ee_goal_cart[:, 2] += 0.05
            elif action == "decrease_eef_goal_y":
                self.teleop_raw_ee_goal_cart[:, 2] -= 0.05
            
            if action == "increase_eef_goal_dr":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 0] += 0.05
            elif action == "decrease_eef_goal_dr":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 0] -= 0.05

            if action == "increse_eef_goal_dp":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 1] += 0.05
            elif action == "decrease_eef_goal_dp":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 1] -= 0.05
            
            if action == "increase_eef_goal_dy":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 2] += 0.05
            elif action == "decrease_eef_goal_dy":
                self.teleop_raw_ee_goal_orn_delta_rpy[:, 2] -= 0.05

        if action == "reset_eef_goal_pose":
            if self.teleop_arm_control_mode == "ee":
                self._reset_teleop_ee_goal_to_default()
                print("[teleop] reset end-effector goal pose to default")

        if action == "open_gripper":
            self.gripper_pos_targets += 0.05
        elif action == "close_gripper":
            self.gripper_pos_targets -= 0.05

        self.gripper_pos_targets = torch.clip(
            self.gripper_pos_targets,
            self.dof_pos_limits[-self.cfg.env.num_gripper_joints:, 0],
            self.dof_pos_limits[-self.cfg.env.num_gripper_joints:, 1],
        )

        self._update_effective_teleop_inputs()

        if self.cfg.goal_ee.sphere_center.mixed_height_reference:
            if action == "set_height_reference_invariant":
                self.goal_height_follow_override = False
                self.goal_height_follow_mask[:] = False
                print("[teleop] height reference mode: z-invariant (obs bit = 0)")
            elif action == "set_height_reference_follow":
                self.goal_height_follow_override = True
                self.goal_height_follow_mask[:] = True
                print("[teleop] height reference mode: trunk-follow (obs bit = 1)")

    def handle_repeated_action(self, action):
        if super().handle_repeated_action(action):
            return True
        if not self.cfg.env.teleop_mode or action not in self.teleop_repeatable_actions:
            return False
        self._apply_teleop_action(action)
        return True

    def handle_viewer_action_event(self, evt):
        super().handle_viewer_action_event(evt)

        if evt.value <= 0:
            return
        self._apply_teleop_action(evt.action)

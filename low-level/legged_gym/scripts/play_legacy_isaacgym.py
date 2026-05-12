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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import euler_from_quat, quat_apply
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import (
    get_load_path,
    apply_checkpoint_features_from_run,
    apply_play_env_schedules_from_metadata,
    configure_playback_curriculum,
    get_run_log_dir,
)

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

def _format_values(values):
    return ", ".join(f"{float(value):.3f}" for value in values)

def _format_contact_flags(flags):
    return "".join("1" if bool(flag) else "0" for flag in flags)

def _print_force_sensor_diagnostics(
    env,
    step_idx,
    summary_interval=100,
    detail_interval=1000,
    support_ratio_warn_high=1.5,
    max_contact_force_warn=300.0,
    body_angle_warn_deg=25.0,
):
    foot_names = list(env.cfg.asset.policy_foot_names)

    sensor_wrench_local_dev = env.force_sensor_tensor[0]
    sensor_wrench_local = sensor_wrench_local_dev.detach().cpu()

    foot_quats = env.rigid_body_state[0, env.feet_indices, 3:7]
    sensor_force_world = quat_apply(
        foot_quats,
        sensor_wrench_local_dev[:, :3],
    ).detach().cpu()

    contact_forces_world = env.contact_forces[0, env.feet_indices, :].detach().cpu()
    foot_state = env.rigid_body_state[0, env.feet_indices, :].detach().cpu()
    foot_vel_world = foot_state[:, 7:10]
    foot_speed = torch.norm(foot_vel_world, dim=-1)

    sensor_contact_flags = env.foot_contacts_from_sensor[0].detach().cpu()
    contact_force_flags = torch.norm(contact_forces_world, dim=-1) > 2.0

    if not hasattr(env, "_diagnostic_total_mass"):
        body_props = env.gym.get_actor_rigid_body_properties(
            env.envs[0],
            env.actor_handles[0],
        )
        env._diagnostic_total_mass = sum(prop.mass for prop in body_props)

    total_mass = float(env._diagnostic_total_mass)
    expected_weight = total_mass * 9.81

    contact_fz = contact_forces_world[:, 2]
    total_vertical_support = float(contact_fz.sum().item())
    support_ratio = total_vertical_support / max(expected_weight, 1e-6)

    front_vertical_support = 0.0
    rear_vertical_support = 0.0
    left_vertical_support = 0.0
    right_vertical_support = 0.0

    for foot_idx, foot_name in enumerate(foot_names):
        fz = float(contact_fz[foot_idx].item())

        if foot_name.startswith("F"):
            front_vertical_support += fz
        elif foot_name.startswith("R"):
            rear_vertical_support += fz

        if "L_" in foot_name or foot_name.startswith("L"):
            left_vertical_support += fz
        elif "R_" in foot_name or foot_name.startswith("R"):
            right_vertical_support += fz

    front_load_ratio = front_vertical_support / max(
        front_vertical_support + rear_vertical_support,
        1e-6,
    )
    left_load_ratio = left_vertical_support / max(
        left_vertical_support + right_vertical_support,
        1e-6,
    )

    root_pos = env.root_states[0, :3].detach().cpu()
    root_lin_vel_body = env.base_lin_vel[0].detach().cpu()
    root_ang_vel_body = env.base_ang_vel[0].detach().cpu()

    roll, pitch, yaw = euler_from_quat(env.base_quat)
    body_rpy_deg = torch.rad2deg(
        torch.stack([roll[0], pitch[0], yaw[0]])
    ).detach().cpu()

    sensor_force_local_norm = torch.norm(sensor_wrench_local[:, :3], dim=-1)
    sensor_force_world_norm = torch.norm(sensor_force_world, dim=-1)
    sensor_torque_norm = torch.norm(sensor_wrench_local[:, 3:], dim=-1)
    sensor_wrench_norm = torch.norm(sensor_wrench_local, dim=-1)
    contact_force_norm = torch.norm(contact_forces_world, dim=-1)

    max_sensor_force_local = float(sensor_force_local_norm.max().item())
    max_sensor_force_world = float(sensor_force_world_norm.max().item())
    max_sensor_wrench = float(sensor_wrench_norm.max().item())
    max_contact_force = float(contact_force_norm.max().item())
    max_contact_force_cfg = float(env.cfg.rewards.max_contact_force)

    alert_items = []
    if support_ratio > support_ratio_warn_high:
        alert_items.append(
            f"support_ratio={support_ratio:.2f}>{support_ratio_warn_high:.2f}"
        )
    if max_contact_force > max_contact_force_warn:
        alert_items.append(
            f"max_contact_force={max_contact_force:.1f}>{max_contact_force_warn:.1f}N"
        )
    wrench_warn = max_contact_force_cfg * support_ratio_warn_high
    if max_sensor_wrench > wrench_warn:
        alert_items.append(
            f"max_sensor_wrench={max_sensor_wrench:.1f}>{wrench_warn:.1f}"
        )
    if abs(float(body_rpy_deg[0].item())) > body_angle_warn_deg:
        alert_items.append(
            f"roll={float(body_rpy_deg[0].item()):.1f}>{body_angle_warn_deg:.1f}deg"
        )
    if abs(float(body_rpy_deg[1].item())) > body_angle_warn_deg:
        alert_items.append(
            f"pitch={float(body_rpy_deg[1].item()):.1f}>{body_angle_warn_deg:.1f}deg"
        )

    has_alert = len(alert_items) > 0

    should_print_summary = (
        step_idx == 0
        or step_idx % summary_interval == 0
    )

    should_print_alert = has_alert and (
        not hasattr(env, "_force_diag_last_alert_step")
        or step_idx - env._force_diag_last_alert_step >= summary_interval
    )

    should_print_detail = (
        step_idx == 0
        or step_idx % detail_interval == 0
    )

    if should_print_alert:
        env._force_diag_last_alert_step = step_idx

    if should_print_summary or should_print_alert:
        alert_text = ""
        if alert_items:
            alert_text = " | alerts=[" + "; ".join(alert_items) + "]"

        print(
            f"[force_diag] step={step_idx:06d} | "
            f"support_ratio={support_ratio:.2f} "
            f"vertical_support={total_vertical_support:.1f}N "
            f"weight={expected_weight:.1f}N | "
            f"front_load_ratio={front_load_ratio:.2f} "
            f"left_load_ratio={left_load_ratio:.2f} | "
            f"max_contact_force={max_contact_force:.1f}N "
            f"max_sensor_force_world={max_sensor_force_world:.1f}N "
            f"max_sensor_wrench={max_sensor_wrench:.1f} | "
            f"sensor_flags={_format_contact_flags(sensor_contact_flags)} "
            f"contact_flags={_format_contact_flags(contact_force_flags)} | "
            f"body_rpy_deg=[{_format_values(body_rpy_deg.tolist())}]"
            f"{alert_text}"
        )

    if not should_print_detail:
        return

    print(
        f"  robot: mass={total_mass:.2f}kg "
        f"configured_max_contact_force={max_contact_force_cfg:.1f} "
        f"max_sensor_force_local={max_sensor_force_local:.1f}N"
    )

    print(
        f"  base: pos=[{_format_values(root_pos.tolist())}] "
        f"lin_vel_body=[{_format_values(root_lin_vel_body.tolist())}] "
        f"ang_vel_body=[{_format_values(root_ang_vel_body.tolist())}]"
    )

    print("  feet:")
    for foot_idx, foot_name in enumerate(foot_names):
        print(
            f"    {foot_name}: "
            f"vertical_force={float(contact_fz[foot_idx].item()):7.1f}N "
            f"contact_force_norm={float(contact_force_norm[foot_idx].item()):7.1f}N "
            f"sensor_force_world_norm={float(sensor_force_world_norm[foot_idx].item()):7.1f}N "
            f"sensor_torque_norm={float(sensor_torque_norm[foot_idx].item()):6.2f} "
            f"sensor_wrench_norm={float(sensor_wrench_norm[foot_idx].item()):7.1f} "
            f"foot_height={float(foot_state[foot_idx, 2].item()):.3f}m "
            f"foot_speed={float(foot_speed[foot_idx].item()):.3f}m/s "
            f"sensor_contact={int(bool(sensor_contact_flags[foot_idx]))} "
            f"contact_force_contact={int(bool(contact_force_flags[foot_idx]))}"
        )


def _apply_static_default_pose(env, env_ids=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    env.root_states[env_ids] = env.base_init_state
    env.root_states[env_ids, :3] += env.env_origins[env_ids]
    env.root_states[env_ids, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))

    env.dof_pos[env_ids] = env.default_dof_pos.unsqueeze(0)
    env.dof_vel[env_ids] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))

    env.actions[env_ids] = 0.0
    env.last_actions[env_ids] = 0.0
    env.last_dof_vel[env_ids] = 0.0
    env.last_torques[env_ids] = 0.0
    env.commands[env_ids] = 0.0
    env.teleop_raw_commands[env_ids] = 0.0
    env.gait_indices[env_ids] = 0.0
    env.gait_frequencies[env_ids] = 0.0
    env.desired_contact_states[env_ids] = 0.0
    env.clock_inputs[env_ids] = 0.0
    env.doubletime_clock_inputs[env_ids] = 0.0
    env.halftime_clock_inputs[env_ids] = 0.0
    env.feet_air_time[env_ids] = 0.0
    env.action_history_buf[env_ids] = 0.0

    env.cfg.env.teleop_mode = True
    env.teleop_arm_control_mode = "joint"
    arm_slice = slice(env.arm_dof_start_idx, env.arm_dof_end_idx)
    gripper_slice = slice(env.num_dofs - env.cfg.env.num_gripper_joints, env.num_dofs)
    env.teleop_arm_joint_pos_targets[env_ids] = env.default_dof_pos[arm_slice]
    env.gripper_pos_targets[env_ids] = env.default_dof_pos[gripper_slice]
    env.teleop_saved_arm_joint_targets[env_ids] = env.teleop_arm_joint_pos_targets[env_ids]
    env.teleop_saved_gripper_pos_targets[env_ids] = env.gripper_pos_targets[env_ids]
    env.teleop_hold_actual_ee_target[env_ids] = False
    env.teleop_initialize_targets_on_next_reset = False

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_jacobian_tensors(env.sim)

def _checkpoint_number_from_path(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def _resolve_playback_checkpoint(log_pth, args, train_cfg):
    checkpoint = int(args.checkpoint)
    if checkpoint == -1:
        checkpoint = int(train_cfg.runner.checkpoint)
    if checkpoint == -1:
        return _checkpoint_number_from_path(get_load_path(log_pth, checkpoint=-1))
    return checkpoint

def play(args):
    log_pth = get_run_log_dir(args.proj_name, args.exptid)
    args, metadata = apply_checkpoint_features_from_run(args, log_pth)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    apply_play_env_schedules_from_metadata(env_cfg, metadata)
    if not args.static_default_pose:
        args.checkpoint = _resolve_playback_checkpoint(log_pth, args, train_cfg)
        if args.curriculum_iter is None:
            args.curriculum_iter = float(args.checkpoint)
    configure_playback_curriculum(env_cfg, args, metadata=metadata)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    # env_cfg.commands.ranges.lin_vel_x = [-1, 1]

    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    if args.episode_length_s is not None:
        env_cfg.env.episode_length_s = float(args.episode_length_s)
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_base_mass = True #False
    env_cfg.domain_rand.randomize_base_com = False

    if args.static_default_pose:
        env_cfg.env.teleop_mode = True
        env_cfg.env.teleop_input_regularization = True
        env_cfg.init_state.rand_yaw_range = 0.0
        env_cfg.init_state.origin_perturb_range = 0.0
        env_cfg.init_state.init_vel_perturb_range = 0.0
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_motor = False
        env_cfg.domain_rand.randomize_gripper_mass = False
        env_cfg.domain_rand.push_robots = False
    
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]

    if args.record_video and args.seed is None:
        args.seed = -1
        env_cfg.seed = -1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    if args.static_default_pose:
        ppo_runner = None
        checkpoint = "static_default_pose"
        policy = None
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
        policy = ppo_runner.get_inference_policy(device=env.device, stochastic=args.stochastic)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if SAVE_ACTOR_HIST_ENCODER:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        model_file = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
        model_name = model_file.split('/')[-1].split('.')[0]
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported')
        os.makedirs(path, exist_ok=True)
        torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), path + '/' + model_name + '_actor.pt')
        print('Saved actor to: ', path + '/' + model_name + '_actor.pt')
    
    if args.use_jit and not args.static_default_pose:
        path = os.path.join(log_pth, 'traced', args.exptid + "_" + str(args.checkpoint) + "_jit.pt")
        print("Loading jit for policy: ", path)
        policy = torch.jit.load(path, map_location=ppo_runner.device)
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    mp4_writers = []
    if args.record_video:
        import imageio
        from datetime import datetime

        env.enable_viewer_sync = False
        video_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(env.num_envs):
            video_name = args.exptid + f'-{i}-' + str(checkpoint) + f"-{video_timestamp}.mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=25)
            mp4_writers.append(mp4_writer)

    if not args.record_video:
        traj_length = 1000*int(env.max_episode_length)
    else:
        traj_length = int(env.max_episode_length)

    # env.update_command_curriculum()
    env.reset()
    if args.static_default_pose:
        _apply_static_default_pose(env)
        obs = env.get_observations()
    for i in range(traj_length):
        start_time = time.time()
        if args.static_default_pose:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        elif args.use_jit:
            actions = policy(torch.cat((obs[:, :env.cfg.env.num_proprio], obs[:, env.cfg.env.num_proprio+env.cfg.env.num_priv:]), dim=1))
        else:
            actions = policy(obs.detach(), hist_encoding=True)
        obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
        if args.static_default_pose and torch.any(dones):
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            _apply_static_default_pose(env, done_env_ids)
            obs = env.get_observations()
        if args.print_force_sensor_every and (i % args.print_force_sensor_every == 0):
            _print_force_sensor_diagnostics(env, i)
        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                for i in range(env.num_envs):
                    mp4_writers[i].append_data(imgs[i])
        
        stop_time = time.time()

        duration = stop_time - start_time
        time.sleep(max(0.02 - duration, 0))

    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()

if __name__ == '__main__':
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

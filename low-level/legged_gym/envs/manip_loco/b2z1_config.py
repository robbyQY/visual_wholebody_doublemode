# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

from .manip_loco_base_config import ManipLocoRoughCfg, ManipLocoRoughCfgPPO


class B2Z1RoughCfg(ManipLocoRoughCfg):
    class goal_ee(ManipLocoRoughCfg.goal_ee):
        class urdf_mount(ManipLocoRoughCfg.goal_ee.urdf_mount):
            arm_base_offset = [0.0, 0.0, 0.0]
            mount_yaw_offset = 0.0
            arm_waist_offset_z = 0.0585
            arm_shoulder_offset_z = 0.045

        class sphere_center(ManipLocoRoughCfg.goal_ee.sphere_center):
            x_offset = 0.25
            y_offset = 0.0
            z_invariant_offset = 0.72
            mixed_height_reference = False
            trunk_follow_ratio = 0.5
            trunk_follow_anchor = "arm_waist"

    class init_state(ManipLocoRoughCfg.init_state):
        pos = [0.0, 0.0, 0.58]
        default_joint_angles = {
            "FL_hip_joint": 0.2,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.2,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.2,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.2,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
            "joint1": 0.0,
            "joint2": 1.48,
            "joint3": -0.63,
            "joint4": -0.84,
            "joint5": 0.0,
            "joint6": 1.57,
            "jointGripper": -0.785,
        }
        rand_yaw_range = np.pi / 2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control(ManipLocoRoughCfg.control):
        stiffness = {
            "FL_hip_joint": 80,
            "FL_thigh_joint": 80,
            "FL_calf_joint": 80,
            "FR_hip_joint": 80,
            "FR_thigh_joint": 80,
            "FR_calf_joint": 80,
            "RL_hip_joint": 80,
            "RL_thigh_joint": 80,
            "RL_calf_joint": 80,
            "RR_hip_joint": 80,
            "RR_thigh_joint": 80,
            "RR_calf_joint": 80,
            "joint1": 5,
            "joint2": 5,
            "joint3": 5,
            "joint4": 5,
            "joint5": 5,
            "joint6": 5,
            "jointGripper": 5,
        }
        damping = {
            "FL_hip_joint": 2.0,
            "FL_thigh_joint": 2.0,
            "FL_calf_joint": 2.0,
            "FR_hip_joint": 2.0,
            "FR_thigh_joint": 2.0,
            "FR_calf_joint": 2.0,
            "RL_hip_joint": 2.0,
            "RL_thigh_joint": 2.0,
            "RL_calf_joint": 2.0,
            "RR_hip_joint": 2.0,
            "RR_thigh_joint": 2.0,
            "RR_calf_joint": 2.0,
            "joint1": 0.5,
            "joint2": 0.5,
            "joint3": 0.5,
            "joint4": 0.5,
            "joint5": 0.5,
            "joint6": 0.5,
            "jointGripper": 0.5,
        }
        adaptive_arm_gains = False
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        decimation = 4
        torque_supervision = False

    class asset(ManipLocoRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
        base_name = "base_link"
        foot_name = "foot"
        gripper_name = "gripper_link"
        arm_waist_name = "joint1"
        hip_joint_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        policy_leg_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        policy_foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        terminate_after_contacts_on = []
        mount_urdf_generator = "b2z1"
        self_collisions = 0
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False

    class arm(ManipLocoRoughCfg.arm):
        init_target_ee_base = [0.25, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class rewards(ManipLocoRoughCfg.rewards):
        reward_scale_preset = "legacy"
        base_height_target = 0.58
        base_height_target_min = 0.33
        base_height_target_max = 0.75
        posture_reference_stand_height = 0.58
        posture_reference_crouch_height = 0.38
        leg_posture_exp_scale = 0.05
        crouch_hip_delta = 0.0
        crouch_thigh_delta = 0.35
        crouch_calf_delta = -0.55

        class scales(ManipLocoRoughCfg.rewards.scales):
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            feet_air_time = 2.0
            feet_height = 1.0
            tracking_lin_vel_max = 2.0
            tracking_lin_vel_x_l1 = 0.0
            tracking_lin_vel_x_exp = 0.0
            tracking_ang_vel = 0.5
            penalty_lin_vel_y = 0.0
            stand_still = 1.0
            stand_still_flexible = 0.0
            walking_dof = 1.5
            walking_dof_flexible = 0.0
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2.0
            pitch = 0.0
            hip_pos = -0.3
            hip_pos_flexible = 0.0
            base_height = -5.0
            base_height_nominal = 0.0
            base_height_band = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            dof_default_pos = 0.0
            dof_error = 0.0
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            action_rate = -0.015
            dof_acc = -7.5e-7
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7 / 4.0
            torques = -2.5e-5
            torques_walking = 0.0
            torques_standing = 0.0
            work = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            ang_vel_xy = -0.2
            collision = -10.0
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001

        class arm_scales(ManipLocoRoughCfg.rewards.arm_scales):
            arm_termination = None
            tracking_ee_sphere = 0.0
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.0
            tracking_ee_orn_ry = None


class B2Z1RoughCfgPPO(ManipLocoRoughCfgPPO):
    class policy(ManipLocoRoughCfgPPO.policy):
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class algorithm(ManipLocoRoughCfgPPO.algorithm):
        torque_supervision = B2Z1RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class runner(ManipLocoRoughCfgPPO.runner):
        experiment_name = "b2z1_v2"

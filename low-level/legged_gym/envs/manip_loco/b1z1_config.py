# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .manip_loco_base_config import ManipLocoRoughCfg, ManipLocoRoughCfgPPO


class B1Z1RoughCfg(ManipLocoRoughCfg):
    class goal_ee(ManipLocoRoughCfg.goal_ee):
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7

        class urdf_mount(ManipLocoRoughCfg.goal_ee.urdf_mount):
            arm_base_offset = [0.3, 0.0, 0.09]
            mount_yaw_offset = 0.0
            arm_waist_offset_z = 0.0585
            arm_shoulder_offset_z = 0.045

        class sphere_center(ManipLocoRoughCfg.goal_ee.sphere_center):
            x_offset = 0.3
            y_offset = 0.0
            z_invariant_offset = 0.7
            mixed_height_reference = False
            trunk_follow_ratio = 0.5
            trunk_follow_anchor = "arm_waist"

    class init_state(ManipLocoRoughCfg.init_state):
        # URDF-backed nominal stance: thigh=0.800, calf=-1.500 gives a
        # geometric contact height of ~0.516 m, so keep the historical 0.500 m spawn.
        pos = [0.0, 0.0, 0.5]
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
            "z1_waist": 0.0,
            "z1_shoulder": 1.48,
            "z1_elbow": -0.63,
            "z1_wrist_angle": -0.84,
            "z1_forearm_roll": 0.0,
            "z1_wrist_rotate": 1.57,
            "z1_jointGripper": -0.785,
        }

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
            "z1_waist": 5,
            "z1_shoulder": 5,
            "z1_elbow": 5,
            "z1_wrist_angle": 5,
            "z1_forearm_roll": 5,
            "z1_wrist_rotate": 5,
            "z1_jointGripper": 5,
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
            "z1_waist": 0.5,
            "z1_shoulder": 0.5,
            "z1_elbow": 0.5,
            "z1_wrist_angle": 0.5,
            "z1_forearm_roll": 0.5,
            "z1_wrist_rotate": 0.5,
            "z1_jointGripper": 0.5,
        }

    class asset(ManipLocoRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf"
        base_name = "trunk"
        gripper_name = "ee_gripper_link"
        arm_waist_name = "z1_waist"
        hip_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        policy_leg_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        policy_foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        penalize_contacts_on = ["thigh", "trunk", "calf"]
        mount_urdf_generator = "b1z1"

    class arm(ManipLocoRoughCfg.arm):
        init_target_ee_base = [0.2, 0.0, 0.2]

    class domain_rand(ManipLocoRoughCfg.domain_rand):
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 15.0]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards(ManipLocoRoughCfg.rewards):
        reward_scale_preset = "legacy"
        base_height_target = 0.55
        base_height_target_min = 0.3
        base_height_target_max = 0.67
        max_contact_force = 40.0
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        leg_posture_exp_scale = 0.05
        crouch_hip_delta = 0.0
        crouch_thigh_delta = 0.35
        crouch_calf_delta = -0.55
        tiptoe_hip_delta = 0.0
        tiptoe_thigh_delta = -0.5565
        tiptoe_calf_delta = 0.8995

        class scales(ManipLocoRoughCfg.rewards.scales):
            tracking_contacts_shaped_force = -2.0 # 惩罚摆动足触地力过大
            tracking_contacts_shaped_vel = -2.0 # 惩罚支撑足滑动过快
            feet_air_time = 2.0 # 奖励迈步腾空更久
            feet_height = 1.0 # 惩罚摆腿抬脚不足
            tracking_lin_vel_max = 2.0 # 奖励前向速度跟踪
            tracking_lin_vel_x_l1 = 0.0 # 奖励前向速度贴近命令
            tracking_lin_vel_x_exp = 0.0 # 奖励前向速度指数跟踪
            tracking_ang_vel = 0.5 # 奖励偏航角速度跟踪
            penalty_lin_vel_y = 0.0 # 惩罚侧向漂移速度
            stand_still = 1.0 # 奖励静止时站稳
            stand_still_flexible = 0.0 # 奖励静止时站稳（Height-flexible）
            walking_dof = 1.5 # 奖励行走时关节规整
            walking_dof_flexible = 0.0 # 奖励行走时关节规整（Height-flexible）
            alive = 1.0 # 奖励回合持续存活
            lin_vel_z = -1.5 # 惩罚机身上下晃动
            roll = -2.0 # 惩罚机身横滚倾斜
            pitch = 0.0 # 惩罚机身俯仰倾斜
            hip_pos = -0.3 # 惩罚髋关节偏离默认
            hip_pos_flexible = 0.0 # 惩罚髋关节偏离默认（Height-flexible）
            base_height = -5.0 # 惩罚机身高度偏差
            base_height_nominal = 0.0 # 在允许区间内弱偏好回到默认高度（Height-flexible）
            base_height_band = 0.0 # 惩罚机身高度偏差（Height-flexible）
            base_height_walking = 0.0 # 惩罚行走时机身高度偏差
            base_height_standing = 0.0 # 惩罚站立时机身高度偏差
            dof_default_pos = 0.0 # 奖励关节贴近默认位
            dof_error = 0.0 # 惩罚关节默认位误差
            orientation = 0.0 # 惩罚机身姿态倾斜
            orientation_walking = 0.0 # 惩罚行走时姿态倾斜
            orientation_standing = 0.0 # 惩罚站立时姿态倾斜
            action_rate = -0.015 # 惩罚动作变化过快
            dof_acc = -7.5e-7 # 惩罚关节加速度过大
            dof_pos_limits = -10.0 # 惩罚关节逼近限位
            delta_torques = -1.0e-7 / 4.0 # 惩罚扭矩突变过大
            torques = -2.5e-5 # 惩罚关节扭矩过大
            torques_walking = 0.0 # 惩罚行走时扭矩过大
            torques_standing = 0.0 # 惩罚站立时扭矩过大
            work = 0.0 # 惩罚腿部净做功大
            energy_square = 0.0 # 惩罚腿部平方能耗
            energy_square_walking = 0.0 # 惩罚行走时平方能耗
            energy_square_standing = 0.0 # 惩罚站立时平方能耗
            ang_vel_xy = -0.2 # 惩罚机身横俯角速度
            collision = -10.0 # 惩罚机身发生碰撞
            feet_jerk = -0.0002 # 惩罚足端冲击突变
            feet_drag = -0.08 # 惩罚足端拖地滑动
            feet_contact_forces = -0.001 # 惩罚足端接触力过大

        class scale_presets(ManipLocoRoughCfg.rewards.scale_presets):
            legacy = {
                "stand_still": 1.0,
                "walking_dof": 1.5,
                "hip_pos": -0.3,
                "base_height": -5.0,
                "stand_still_flexible": 0.0,
                "walking_dof_flexible": 0.0,
                "pitch": 0.0,
                "hip_pos_flexible": 0.0,
                "base_height_nominal": 0.0,
                "base_height_band": 0.0,
                "orientation": 0.0,
                "ang_vel_xy": -0.2,
            }
            height_flexible = {
                "stand_still": 0.0,
                "walking_dof": 0.0,
                "hip_pos": 0.0,
                "base_height": 0.0,
                "stand_still_flexible": 1.0,
                "walking_dof_flexible": 1.5,
                "pitch": -2.0,
                "hip_pos_flexible": -0.1,
                "base_height_nominal": -0.25,
                "base_height_band": -5.0,
                "orientation": -6.0,
                "ang_vel_xy": -0.4,
            }

        _selected_reward_scale_preset = getattr(scale_presets, reward_scale_preset, None)
        if _selected_reward_scale_preset is None:
            raise ValueError(f"Unsupported rewards.reward_scale_preset={reward_scale_preset}")
        for _reward_name, _reward_scale in _selected_reward_scale_preset.items():
            setattr(scales, _reward_name, _reward_scale)
        del _selected_reward_scale_preset, _reward_name, _reward_scale

        class arm_scales(ManipLocoRoughCfg.rewards.arm_scales):
            arm_termination = None # 惩罚机械臂回合终止
            tracking_ee_sphere = 0.0 # 奖励末端球坐标跟踪
            tracking_ee_world = 0.8 # 奖励末端世界坐标跟踪
            tracking_ee_sphere_walking = 0.0 # 奖励行走时末端球坐标跟踪
            tracking_ee_sphere_standing = 0.0 # 奖励站立时末端球坐标跟踪
            tracking_ee_cart = None # 奖励末端笛卡尔跟踪
            arm_energy_abs_sum = None # 惩罚机械臂能耗过大
            tracking_ee_orn = 0.0 # 奖励末端姿态跟踪
            tracking_ee_orn_ry = None # 奖励末端滚偏姿态跟踪


class B1Z1RoughCfgPPO(ManipLocoRoughCfgPPO):
    class policy(ManipLocoRoughCfgPPO.policy):
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains

    class algorithm(ManipLocoRoughCfgPPO.algorithm):
        torque_supervision = B1Z1RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains

    class runner(ManipLocoRoughCfgPPO.runner):
        experiment_name = "b1z1_v2"

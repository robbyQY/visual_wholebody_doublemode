# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .manip_loco_base_config import ManipLocoRoughCfg, ManipLocoRoughCfgPPO


class B2Z1RoughCfg(ManipLocoRoughCfg):
    class goal_ee(ManipLocoRoughCfg.goal_ee):
        collision_upper_limits = [0.2, 0.2, -0.05]
        collision_lower_limits = [-0.7, -0.2, -0.7]
        underground_limit = -0.7

        class urdf_mount(ManipLocoRoughCfg.goal_ee.urdf_mount):
            arm_base_offset = [0.2, 0.0, 0.09]
            mount_yaw_offset = 0.0
            arm_waist_offset_z = 0.0585
            arm_shoulder_offset_z = 0.045

        class sphere_center(ManipLocoRoughCfg.goal_ee.sphere_center):
            x_offset = 0.2
            y_offset = 0.0
            z_invariant_offset = 0.7
            mixed_height_reference = False
            trunk_follow_ratio = 0.5
            trunk_follow_anchor = "arm_waist"

    class init_state(ManipLocoRoughCfg.init_state):
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
            "joint1": 0.0,
            "joint2": 1.48,
            "joint3": -0.63,
            "joint4": -0.84,
            "joint5": 0.0,
            "joint6": 1.57,
            "jointGripper": -0.785,
        }

    class control(ManipLocoRoughCfg.control):
        stiffness = {
            "FL_hip_joint": 100,
            "FL_thigh_joint": 100,
            "FL_calf_joint": 100,
            "FR_hip_joint": 100,
            "FR_thigh_joint": 100,
            "FR_calf_joint": 100,
            "RL_hip_joint": 100,
            "RL_thigh_joint": 100,
            "RL_calf_joint": 100,
            "RR_hip_joint": 100,
            "RR_thigh_joint": 100,
            "RR_calf_joint": 100,
            "joint1": 5,
            "joint2": 5,
            "joint3": 5,
            "joint4": 5,
            "joint5": 5,
            "joint6": 5,
            "jointGripper": 5,
        }
        damping = {
            "FL_hip_joint": 3.0,
            "FL_thigh_joint": 3.0,
            "FL_calf_joint": 3.0,
            "FR_hip_joint": 3.0,
            "FR_thigh_joint": 3.0,
            "FR_calf_joint": 3.0,
            "RL_hip_joint": 3.0,
            "RL_thigh_joint": 3.0,
            "RL_calf_joint": 3.0,
            "RR_hip_joint": 3.0,
            "RR_thigh_joint": 3.0,
            "RR_calf_joint": 3.0,
            "joint1": 0.5,
            "joint2": 0.5,
            "joint3": 0.5,
            "joint4": 0.5,
            "joint5": 0.5,
            "joint6": 0.5,
            "jointGripper": 0.5,
        }

    class asset(ManipLocoRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
        base_name = "base_link"
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
        mount_urdf_generator = "b2z1"

    class arm(ManipLocoRoughCfg.arm):
        init_target_ee_base = [0.2, 0.0, 0.2]

    class domain_rand(ManipLocoRoughCfg.domain_rand):
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 4.0]
        randomize_base_com = True
        added_com_range_x = [-0.03, 0.03]
        added_com_range_y = [-0.03, 0.03]
        added_com_range_z = [-0.03, 0.03]
        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]
        arm_motor_strength_range = [0.9, 1.1]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0.0, 0.02]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards(ManipLocoRoughCfg.rewards):
        reward_scale_preset = "legacy"

        # legacy 和 height_flexible 模式的参数
        base_height_target = 0.55
        base_height_target_min = 0.3
        base_height_target_max = 0.67
        max_contact_force = 55.0
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        gait_transition_lower = 0.1
        gait_transition_upper = 0.9
        leg_posture_exp_scale = 0.05
        crouch_hip_delta = 0.0
        crouch_thigh_delta = 0.35
        crouch_calf_delta = -0.55
        tiptoe_hip_delta = 0.0
        tiptoe_thigh_delta = -0.5565
        tiptoe_calf_delta = 0.8995

        # robot_lab Unitree B2 模式的参数
        robotlab_tracking_std = 0.5
        robotlab_undesired_contacts_threshold = 1.0
        robotlab_contact_force_threshold = 100.0
        robotlab_stand_still_scale = 5.0
        robotlab_velocity_threshold = 0.5
        robotlab_command_threshold = 0.1
        robotlab_feet_height_body_target = -0.4
        robotlab_feet_height_tanh_mult = 2.0

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

            robotlab_lin_vel_z_l2 = 0.0 # robot_lab：惩罚机身 z 方向线速度平方
            robotlab_ang_vel_xy_l2 = 0.0 # robot_lab：惩罚机身横滚/俯仰角速度平方
            robotlab_joint_torques_l2 = 0.0 # robot_lab：惩罚关节扭矩平方
            robotlab_joint_acc_l2 = 0.0 # robot_lab：惩罚关节加速度平方
            robotlab_joint_pos_limits = 0.0 # robot_lab：惩罚关节接近或超过位置限位
            robotlab_joint_power = 0.0 # robot_lab：惩罚关节功率绝对值 |tau * qd|
            robotlab_stand_still = 0.0 # robot_lab：低速命令时惩罚关节偏离默认位
            robotlab_joint_pos_penalty = 0.0 # robot_lab：按行走/静止状态惩罚关节偏离默认位
            robotlab_joint_mirror = 0.0 # robot_lab：惩罚对角腿关节姿态不镜像
            robotlab_action_rate_l2 = 0.0 # robot_lab：惩罚动作变化平方
            robotlab_undesired_contacts = 0.0 # robot_lab：惩罚非足端刚体发生接触
            robotlab_contact_forces = 0.0 # robot_lab：惩罚足端接触力超过阈值
            robotlab_track_lin_vel_xy_exp = 0.0 # robot_lab：奖励 xy 平面线速度指数跟踪
            robotlab_track_ang_vel_z_exp = 0.0 # robot_lab：奖励偏航角速度指数跟踪
            robotlab_feet_contact_without_cmd = 0.0 # robot_lab：无运动命令时奖励足端接触
            robotlab_feet_height_body = 0.0 # robot_lab：惩罚足端在机体系下偏离目标高度
            robotlab_upward = 0.0 # robot_lab：奖励机身朝上
            robotlab_clock_swing_force = 0.0 # robot_lab-clock：跟随clock，惩罚摆动足触地力过大
            robotlab_clock_stance_vel = 0.0 # robot_lab-clock：跟随clock，惩罚支撑足滑动过快
            robotlab_clock_stance_contact = 0.0 # robot_lab-clock：跟随clock，轻微惩罚支撑足接触力不足

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
            robotlab_b2 = {
                "tracking_contacts_shaped_force": 0.0,
                "tracking_contacts_shaped_vel": 0.0,
                "feet_air_time": 0.0,
                "feet_height": 0.0,
                "tracking_lin_vel_max": 0.0,
                "tracking_lin_vel_x_l1": 0.0,
                "tracking_lin_vel_x_exp": 0.0,
                "tracking_ang_vel": 0.0,
                "penalty_lin_vel_y": 0.0,
                "stand_still": 0.0,
                "stand_still_flexible": 0.0,
                "walking_dof": 0.0,
                "walking_dof_flexible": 0.0,
                "alive": 0.0,
                "lin_vel_z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "hip_pos": 0.0,
                "hip_pos_flexible": 0.0,
                "base_height": 0.0,
                "base_height_nominal": 0.0,
                "base_height_band": 0.0,
                "base_height_walking": 0.0,
                "base_height_standing": 0.0,
                "dof_default_pos": 0.0,
                "dof_error": 0.0,
                "orientation": 0.0,
                "orientation_walking": 0.0,
                "orientation_standing": 0.0,
                "action_rate": 0.0,
                "dof_acc": 0.0,
                "dof_pos_limits": 0.0,
                "delta_torques": 0.0,
                "torques": 0.0,
                "torques_walking": 0.0,
                "torques_standing": 0.0,
                "work": 0.0,
                "energy_square": 0.0,
                "energy_square_walking": 0.0,
                "energy_square_standing": 0.0,
                "ang_vel_xy": 0.0,
                "collision": 0.0,
                "feet_jerk": 0.0,
                "feet_drag": 0.0,
                "feet_contact_forces": 0.0,

                "robotlab_lin_vel_z_l2": -2.0,
                "robotlab_ang_vel_xy_l2": -0.05,
                "robotlab_joint_torques_l2": -1e-5,
                "robotlab_joint_acc_l2": -1e-7,
                "robotlab_joint_pos_limits": -5.0,
                "robotlab_joint_power": -1e-5,
                "robotlab_stand_still": -2.0,
                "robotlab_joint_pos_penalty": -1.0,
                "robotlab_joint_mirror": -0.05,
                "robotlab_action_rate_l2": -0.01,
                "robotlab_undesired_contacts": -1.0,
                "robotlab_contact_forces": -1.5e-4,
                "robotlab_track_lin_vel_xy_exp": 3.0,
                "robotlab_track_ang_vel_z_exp": 1.5,
                "robotlab_feet_contact_without_cmd": 0.1,
                "robotlab_feet_height_body": -5.0,
                "robotlab_upward": 3.0,
                "robotlab_clock_swing_force": -2.0,
                "robotlab_clock_stance_vel": -2.0,
                "robotlab_clock_stance_contact": -0.5,
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


class B2Z1RoughCfgPPO(ManipLocoRoughCfgPPO):
    class policy(ManipLocoRoughCfgPPO.policy):
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class algorithm(ManipLocoRoughCfgPPO.algorithm):
        torque_supervision = B2Z1RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B2Z1RoughCfg.control.adaptive_arm_gains

    class runner(ManipLocoRoughCfgPPO.runner):
        experiment_name = "b2z1_v2"

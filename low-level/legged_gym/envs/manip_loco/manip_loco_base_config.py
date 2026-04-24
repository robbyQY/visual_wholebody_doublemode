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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


class ManipLocoRoughCfg(LeggedRobotCfg):
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        num_collision_check_samples = 10
        command_mode = "sphere"
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p

        class urdf_mount:
            # Keep shared URDF-derived arm mounting parameters together. Robot-specific mount positions live in each task config.
            mount_yaw_offset = 0.0
            arm_waist_offset_z = 0.0585
            arm_shoulder_offset_z = 0.045

        class sphere_center:
            y_offset = 0.0 # Relative to base
            mixed_height_reference = False # If true, train both z-invariant and trunk-height-following goal modes, with a mode bit in observations
            trunk_follow_ratio = 0.5 # Fraction of trunk-height-following goal episodes when mixed_height_reference is enabled
            trunk_follow_anchor = "arm_waist" # arm_base | arm_waist | arm_shoulder

        class ranges:
            init_pos_start = [0.5, np.pi / 8, 0]
            init_pos_end = [0.7, 0, 0]
            pos_l = [0.4, 0.95]
            pos_p = [-1 * np.pi / 2.1, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]  # relative yaw offset from the episode start yaw when omnidirectional_pos_y is enabled
            omnidirectional_pos_y = False
            omnidirectional_init_pos_y = [-5 * np.pi / 6, 5 * np.pi / 6]  # reset-time absolute yaw range before intersecting with the Z1 waist joint limits
            omnidirectional_rear_transition_pos_y_abs = 2 * np.pi / 3  # keep front/side ranges unchanged until |yaw| exceeds 120 deg
            omnidirectional_pos_l = [0.2, 0.95]  # omni-only radial range before the rear-sector contraction kicks in
            omnidirectional_rear_pos_l = [0.2, 0.5]  # omni-only radial range at directly behind the robot
            omnidirectional_rear_pos_p = [-np.pi / 6, np.pi / 6]  # omni-only pitch range at directly behind the robot
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.0 # time before command are changed[s]

        # Command-range curricula
        lin_vel_x_min_schedule = [0.0, -0.8, 5000, 5000]
        lin_vel_x_max_schedule = [0.8, 0.8, 0, 0]
        ang_vel_yaw_schedule = [1.0, 1.0, 0, 0]
        non_omni_pos_y_schedule = [1.2, 1.2, 0, 0]
        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 6144
        num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = 3  # -1 for no delay
        action_delay_mode = "auto"  # auto: keep training curriculum, undelayed: latest action, delayed: one-step delayed action
        ee_goal_obs_mode = "command"  # command: use sampled EE command directly, arm_base_target: use target relative to arm base
        num_gripper_joints = 1
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        num_priv = 5 + 1 + 12
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds
        reorder_dofs = True
        teleop_mode = False # Overriden in teleop.py. When true, commands come from keyboard
        teleop_input_regularization = False # If true, preprocess teleop inputs before feeding the policy/control stack
        teleop_zero_lin_vel_x_clip = 0.2
        teleop_zero_ang_vel_yaw_clip = 0.5
        teleop_lin_vel_x_limit = 0.8
        teleop_ang_vel_yaw_limit = 1.0
        teleop_ee_goal_x_limit = [-0.5, 1.0]
        teleop_ee_goal_y_limit = [-0.7, 0.7]
        teleop_ee_goal_z_limit = [-0.6, 0.6]
        teleop_restore_arm_gripper_state_on_reset = False
        teleop_key_repeat_delay_s = 0.35
        teleop_key_repeat_rate_hz = 6.0
        record_video = False
        stand_by = False
        observe_gait_commands = False
        gait_frequency_min = 2.0
        gait_frequency_max = 2.0
        gait_frequency_lin_vel_ref = 1.2
        gait_frequency_ang_vel_ref = 2.0
        gait_frequency_ang_vel_weight = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = {} # = target angles [rad] when action = 0.0
        rand_yaw_range = np.pi / 2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        stiffness = {}  # [N*m/rad] # Keys can be exact joint names or substrings.
        damping = {}     # [N*m*s/rad]
        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 4 + [2.1, 0.6, 0.6, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset(LeggedRobotCfg.asset):
        foot_name = "foot"
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse=\"true\">"
        fix_base_link = False
        robot_ablation = "none"
        leg_collision_scale = 1.0

    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16

    class arm:
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = True
        randomize_base_com = True
        randomize_motor = True
        randomize_gripper_mass = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards:
        reward_container_name = "maniploco_rewards"   # select the reward container to use

        # -------Common Para. ---------
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        # -------Gait control Para. ---------
        kappa_gait_probs = 0.07
        gait_transition_lower = 0.1
        gait_transition_upper = 0.9
        feet_height_target = 0.3
        feet_airtime_allfeet = False
        feet_height_allfeet = False

        class scales:
            # Reward coefficients are robot-specific and live in each task config.
            pass

        class scale_presets:
            # Preset maps are also task-specific and live in each task config.
            legacy = {}
            height_flexible = {}

        class arm_scales:
            # Arm reward coefficients are task-specific and live in each task config.
            pass

    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = "trimesh" # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.1] # [0.04, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)
        terrain_dict = {
            "smooth slope": 0.0,
            "rough slope up": 0.0,
            "rough slope down": 0.0,
            "rough stairs up": 0.0,
            "rough stairs down": 0.0,
            "discrete": 0.0,
            "stepping stones": 0.0,
            "gaps": 0.0,
            "rough flat": 1.0,
            "pit": 0.0,
            "wall": 0.0,
        }
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False


class ManipLocoRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = "elu" # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_tanh = False
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        priv_encoder_dims = [64, 20]
        num_leg_actions = 12
        num_arm_actions = 6
        adaptive_arm_gains = ManipLocoRoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4
        schedule = "fixed" # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]
        mixing_schedule = [1.0, 0, 3000] #if not RESUME else [1.0, 0, 1]
        torque_supervision = ManipLocoRoughCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = ManipLocoRoughCfg.control.adaptive_arm_gains
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedule = [0, 0.1, 3000, 10000]

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 45000 # number of policy updates
        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = "manip_loco_v2"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

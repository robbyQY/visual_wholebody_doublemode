from isaaclab.utils import configclass
from legged_gym.envs.base.legged_robot_config import LeggedRobotIsaacLabCfg, LegacyControlCfg


@configclass
class B2Z1IsaacLabCfg(LeggedRobotIsaacLabCfg):
    action_space = 18
    observation_space = 810
    num_proprio = 72
    num_priv = 18
    history_len = 10
    observe_gait_commands = True
    mixed_height_reference = True
    action_delay = 3
    action_delay_mode = "undelayed"
    num_gripper_joints = 1

    episode_length_s = 20.0
    decimation = 4
    base_body_name = "base_link"
    gripper_body_name = "gripper_link"
    num_gripper_joints = 1
    policy_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # terminate_body_names = ["base_link", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    terminate_body_names = []
    default_joint_angles = {
        "FL_hip_joint": 0.2, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "FR_hip_joint": -0.2, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "RL_hip_joint": 0.2, "RL_thigh_joint": 0.8, "RL_calf_joint": -1.5,
        "RR_hip_joint": -0.2, "RR_thigh_joint": 0.8, "RR_calf_joint": -1.5,
        "joint1": 0.0, "joint2": 1.48, "joint3": -0.63, "joint4": -0.84,
        "joint5": 0.0, "joint6": 1.57, "jointGripper": -0.785,
    }
    control = LegacyControlCfg(
        control_type="P",
        # action_scale=0.5,
        action_scale=[
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            0.4, 0.45, 0.45,
            2.1, 0.6, 0.6,
            0.0, 0.0, 0.0,
        ],
        clip_actions=100.0,
        clip_observations=100.0,
        stiffness={
            "FL_hip_joint": 100, "FL_thigh_joint": 100, "FL_calf_joint": 100,
            "FR_hip_joint": 100, "FR_thigh_joint": 100, "FR_calf_joint": 100,
            "RL_hip_joint": 100, "RL_thigh_joint": 100, "RL_calf_joint": 100,
            "RR_hip_joint": 100, "RR_thigh_joint": 100, "RR_calf_joint": 100,
            "joint1": 5, "joint2": 5, "joint3": 5, "joint4": 5, "joint5": 5, "joint6": 5,
            "jointGripper": 5,
        },
        damping={
            "FL_hip_joint": 3.0, "FL_thigh_joint": 3.0, "FL_calf_joint": 3.0,
            "FR_hip_joint": 3.0, "FR_thigh_joint": 3.0, "FR_calf_joint": 3.0,
            "RL_hip_joint": 3.0, "RL_thigh_joint": 3.0, "RL_calf_joint": 3.0,
            "RR_hip_joint": 3.0, "RR_thigh_joint": 3.0, "RR_calf_joint": 3.0,
            "joint1": 0.5, "joint2": 0.5, "joint3": 0.5, "joint4": 0.5, "joint5": 0.5, "joint6": 0.5,
            "jointGripper": 0.5,
        },
        # stiffness={
        #     "FL_hip_joint": 180, "FL_thigh_joint": 180, "FL_calf_joint": 180,
        #     "FR_hip_joint": 180, "FR_thigh_joint": 180, "FR_calf_joint": 180,
        #     "RL_hip_joint": 180, "RL_thigh_joint": 180, "RL_calf_joint": 180,
        #     "RR_hip_joint": 180, "RR_thigh_joint": 180, "RR_calf_joint": 180,

        #     "joint1": 20,
        #     "joint2": 80,
        #     "joint3": 80,
        #     "joint4": 30,
        #     "joint5": 20,
        #     "joint6": 20,
        #     "jointGripper": 10,
        # },
        # damping={
        #     "FL_hip_joint": 6.0, "FL_thigh_joint": 6.0, "FL_calf_joint": 6.0,
        #     "FR_hip_joint": 6.0, "FR_thigh_joint": 6.0, "FR_calf_joint": 6.0,
        #     "RL_hip_joint": 6.0, "RL_thigh_joint": 6.0, "RL_calf_joint": 6.0,
        #     "RR_hip_joint": 6.0, "RR_thigh_joint": 6.0, "RR_calf_joint": 6.0,

        #     "joint1": 2.0,
        #     "joint2": 4.0,
        #     "joint3": 4.0,
        #     "joint4": 2.0,
        #     "joint5": 1.0,
        #     "joint6": 1.0,
        #     "jointGripper": 1.0,
        # },        
    )

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
        episode_length_s = 30 # episode length in seconds
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
        observe_gait_commands = True
        gait_frequency_min = 2.0
        gait_frequency_max = 2.0
        gait_frequency_lin_vel_ref = 1.2
        gait_frequency_ang_vel_ref = 2.0
        gait_frequency_ang_vel_weight = 1.0

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

    class asset():
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
        
    def __post_init__(self):
        super().__post_init__()

        # IsaacLab only reads robot.init_state.joint_pos for initial joint positions.
        # Our legacy field default_joint_angles is only for migrated control logic.
        self.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # IsaacLab uses wxyz
        # self.robot.init_state.rot = (0.70710678, -0.70710678, 0.0, 0.0)  # IsaacLab uses wxyz 
        self.robot.init_state.joint_pos = dict(self.default_joint_angles)    

from isaaclab.utils import configclass
from legged_gym.envs.base.legged_robot_isaaclab_config import LeggedRobotIsaacLabCfg, LegacyControlCfg


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
    
    def __post_init__(self):
        super().__post_init__()

        # IsaacLab only reads robot.init_state.joint_pos for initial joint positions.
        # Our legacy field default_joint_angles is only for migrated control logic.
        self.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # IsaacLab uses wxyz
        # self.robot.init_state.rot = (0.70710678, -0.70710678, 0.0, 0.0)  # IsaacLab uses wxyz 
        self.robot.init_state.joint_pos = dict(self.default_joint_angles)    

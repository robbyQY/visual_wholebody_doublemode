from __future__ import annotations
import os
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class LegacyControlCfg:
    control_type: str = "P"
    action_scale: float = 0.5
    stiffness: dict = MISSING
    damping: dict = MISSING
    clip_actions: float = 100.0
    clip_observations: float = 100.0


@configclass
class LegacyCommandCfg:
    num_commands: int = 3
    resampling_time: float = 3.0
    lin_vel_x: tuple = (-0.8, 0.8)
    lin_vel_y: tuple = (0.0, 0.0)
    ang_vel_yaw: tuple = (-1.0, 1.0)


@configclass
class LeggedRobotIsaacLabCfg(DirectRLEnvCfg):
    """Minimal DirectRLEnvCfg generated from the uploaded legged_gym configs.

    Fill `robot_usd_path` with a URDF-converted USD. IsaacLab uses USD assets;
    the old gym.load_asset(URDF) path is intentionally not used here.
    """
    # RL timing
    decimation: int = 4
    episode_length_s: float = 20.0
    action_space: int = 12
    observation_space: int = 235
    state_space: int = 0

    # simulation / scene
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # terrain: use plane first. Rebuild rough terrain after policy I/O is verified.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    # Asset path. Override with --robot_urdf_path or edit here.
    robot_urdf_path: str = os.environ.get(
        "LEGGED_GYM_ROBOT_URDF",
        "/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/b2z1_isaacsim_mesh_axis_fixed.urdf",
        # "/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/b2z1.urdf",
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="",  # patched at runtime from cfg.robot_urdf_path
            activate_contact_sensors=True,
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=True,
            self_collision=False,
            make_instanceable=False,            
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),    
            ),        
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}),
        actuators={
            # Legs: old pipeline uses manual torque PD.
            # Keep stiffness/damping 0 to avoid double PD.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
                effort_limit_sim=600.0,
                velocity_limit_sim=100.0,
                stiffness=0,
                damping=0,
            ),            
            # "legs": ImplicitActuatorCfg(
            #     joint_names_expr=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
            #     effort_limit_sim=600.0,
            #     velocity_limit_sim=100.0,
            #     stiffness={
            #         ".*hip_joint": 500.0,
            #         ".*thigh_joint": 500.0,
            #         ".*calf_joint": 500.0,
            #     },
            #     damping={
            #         ".*hip_joint": 20.0,
            #         ".*thigh_joint": 20.0,
            #         ".*calf_joint": 20.0,
            #     },
            # ),

            # Z1 arm + gripper: old ManipLoco used position target drive.
            # Give IsaacLab drive stiffness/damping here.
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
                ],
                effort_limit_sim=80.0,
                velocity_limit_sim=20.0,
                stiffness={
                    "joint1": 20.0,
                    "joint2": 80.0,
                    "joint3": 80.0,
                    "joint4": 30.0,
                    "joint5": 20.0,
                    "joint6": 20.0,
                    "jointGripper": 10.0,
                },
                damping={
                    "joint1": 2.0,
                    "joint2": 4.0,
                    "joint3": 4.0,
                    "joint4": 2.0,
                    "joint5": 1.0,
                    "joint6": 1.0,
                    "jointGripper": 1.0,
                },
            ),
        }
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        # prim_path="/World/envs/env_.*/Robot/.*",
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
    )

    # legacy-like fields used by environment code
    control: LegacyControlCfg = LegacyControlCfg(stiffness={}, damping={})
    commands: LegacyCommandCfg = LegacyCommandCfg()
    default_joint_angles: dict = MISSING
    policy_joint_names: list = MISSING
    foot_body_names: list = MISSING
    terminate_body_names: list = MISSING
    base_body_name: str = "base_link"
    gripper_body_name: str = "gripper_link"
    num_gripper_joints: int = 1
    send_timeouts: bool = True
    enable_height_scan: bool = False

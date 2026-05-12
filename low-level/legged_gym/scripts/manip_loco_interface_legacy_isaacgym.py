from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import euler_from_quat
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path, apply_checkpoint_features_from_run, get_run_log_dir

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

B1_FREQ = 50
B1_STEP_TIME = 1. / B1_FREQ
LOW_HIGH_RATE = 5
Z1_FREQ = 500

LIN_VEL_X_CLIP = 0.15
ANG_VEL_YAW_CLIP = 0.3
ANG_VEL_PITCH_CLIP = ANG_VEL_YAW_CLIP

GAIT_WAIT_TIME = 35


class ManipLoco_Policy():
    def __init__(self, args) -> None:
        self.args = args
        self.env = None
        self.policy = None
        self.obs = None
        self.env_cfg = None
        self.init_env()
        self.init_logger()
        self.timestamp = 0

    def _format_vec(self, tensor, env_id=0):
        values = tensor[env_id].detach().cpu().tolist()
        return [round(v, 3) for v in values]

    def _format_pose6(self, pos_tensor, rpy_tensor, env_id=0):
        pos = pos_tensor[env_id].detach().cpu().tolist()
        rpy = rpy_tensor[env_id].detach().cpu().tolist()
        return [round(v, 3) for v in pos + rpy]

    def init_logger(self):
        logger = Logger(self.env.dt)
        robot_index = 0  # which robot is used for logging
        joint_index = 1  # which joint is used for logging
        stop_state_log = 100  # number of steps before plotting states
        stop_rew_log = self.env.max_episode_length + 1  # number of steps before print average episode rewards
        camera_position = np.array(self.env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(self.env_cfg.viewer.lookat) - np.array(self.env_cfg.viewer.pos)
        img_idx = 0

    def init_env(self):
        log_pth = get_run_log_dir(self.args.proj_name, self.args.exptid)
        self.args, _ = apply_checkpoint_features_from_run(self.args, log_pth)
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        # override some parameters for testing
        env_cfg.env.num_envs = 1
        env_cfg.env.teleop_mode = True
        env_cfg.env.episode_length_s = 10000
        env_cfg.domain_rand.push_robots = False
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 3

        self.env_cfg = env_cfg

        # prepare environment
        self.env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=env_cfg)

        # initial observation
        self.obs = self.env.get_observations()

        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(
            log_root=log_pth,
            env=self.env,
            name=self.args.task,
            args=self.args,
            train_cfg=train_cfg,
            return_log_dir=True,
        )
        self.policy = ppo_runner.get_inference_policy(device=self.env.device, stochastic=self.args.stochastic)

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print("Exported policy as jit script to: ", path)

        if SAVE_ACTOR_HIST_ENCODER:
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
            model_file = get_load_path(log_root, load_run=self.args.load_run, checkpoint=self.args.checkpoint)
            model_name = model_file.split("/")[-1].split(".")[0]
            path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                train_cfg.runner.load_run,
                "exported",
            )
            os.makedirs(path, exist_ok=True)
            torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), path + "/" + model_name + "_actor.pt")
            print("Saved actor to: ", path + "/" + model_name + "_actor.pt")

        if self.args.use_jit:
            path = os.path.join(log_pth, "traced", self.args.exptid + "_" + str(checkpoint) + "_jit.pt")
            print("Loading jit for policy: ", path)
            self.policy = torch.jit.load(path, map_location=ppo_runner.device)

    def step(self):
        start_time = time.time()
        obs = self.obs

        if args.use_jit:
            actions = self.policy(torch.cat((obs[:, :self.env.cfg.env.num_proprio], obs[:, self.env.cfg.env.num_priv:]), dim=1))
        else:
            actions = self.policy(obs.detach(), hist_encoding=True)

        self.obs, _, rews, arm_rews, dones, infos = self.env.step(actions.detach())

        if self.timestamp % 10 == 0:
            sim_time = (self.timestamp + 1) * self.env.dt
            arm_mode = getattr(self.env, "teleop_arm_control_mode", "ee")
            arm_joint_slice = slice(self.env.arm_dof_start_idx, self.env.arm_dof_end_idx)
            current_ee_rpy = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
            command_ee_6dof = self._format_pose6(self.env.curr_ee_goal_cart_world, self.env.ee_goal_orn_euler)
            current_ee_6dof = self._format_pose6(self.env.ee_pos, current_ee_rpy)
            current_arm_joint = self._format_vec(self.env.dof_pos[:, arm_joint_slice])

            command_line = (
                f"[sim_time={sim_time:.2f}s] [command] [arm_mode={arm_mode}] "
                f"vel_cmd={self.env.commands[0, 0].item():.3f}, "
                f"yaw_cmd={self.env.commands[0, 2].item():.3f}, "
            )
            if arm_mode == "joint":
                command_line += f"arm_joint_cmd={self._format_vec(self.env.teleop_arm_joint_pos_targets)}"
            else:
                command_line += f"ee_cmd_6dof={command_ee_6dof}"

            state_line = (
                f"[sim_time={sim_time:.2f}s] [state] "
                f"vel={self.env.base_lin_vel[0, 0].item():.3f}, "
                f"yaw={self.env.base_ang_vel[0, 2].item():.3f}, "
                f"ee_6dof={current_ee_6dof}, "
                f"arm_joints={current_arm_joint}"
            )
            print(command_line)
            print(state_line)

        stop_time = time.time()
        duration = stop_time - start_time
        time.sleep(max(0.02 - duration, 0))

        self.timestamp += 1


if __name__ == "__main__":
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    args = get_args()
    manipLoco = ManipLoco_Policy(args)
    while True:
        manipLoco.step()

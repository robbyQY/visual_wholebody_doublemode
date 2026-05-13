"""Minimal IsaacLab play script for the migrated B2Z1 DirectRLEnv.
Run with IsaacLab's python, e.g.:
  ./isaaclab.sh -p legged_gym/scripts/play_isaaclab.py --robot_usd_path /abs/path/b2z1.usd --num_envs 1
"""
from __future__ import annotations

import sys
from pathlib import Path

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
if str(LOW_LEVEL_ROOT) in sys.path:
    sys.path.remove(str(LOW_LEVEL_ROOT))
sys.path.insert(0, str(LOW_LEVEL_ROOT))

print("[play_isaaclab] Using LOW_LEVEL_ROOT:", LOW_LEVEL_ROOT)
import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot_urdf_path",
    type=str,
    # default="/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/b2z1_isaacsim_mesh_axis_fixed.urdf",
    default="/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/b2z1.urdf",
)
parser.add_argument("--num_envs", type=int, default=1)
# parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from legged_gym.envs.manip_loco.b2z1_isaaclab_config import B2Z1IsaacLabCfg
from legged_gym.envs.manip_loco.manip_loco_isaaclab import ManipLocoIsaacLab

cfg = B2Z1IsaacLabCfg()
cfg.robot_urdf_path = args.robot_urdf_path
cfg.robot.spawn.asset_path = args.robot_urdf_path
cfg.scene.num_envs = args.num_envs

env = ManipLocoIsaacLab(cfg)
obs, _ = env.reset()
print("joint_names:", env.robot.data.joint_names)
print("body_names:", env.robot.data.body_names)

while simulation_app.is_running():
    with torch.inference_mode():
        actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        obs, rew, terminated, truncated, info = env.step(actions)
        obs_dict = env.get_observations()
        obs = obs_dict["policy"]
        print(obs.shape)
        # print(env.robot.data.root_pos_w[0])
        # print(env.robot.data.root_lin_vel_w[0])
        # print(obs[0, :20])

env.close()
simulation_app.close()

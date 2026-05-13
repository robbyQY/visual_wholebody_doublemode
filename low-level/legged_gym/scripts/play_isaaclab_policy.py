"""IsaacLab policy play script for migrated B2Z1 DirectRLEnv.

This script keeps the old RSL-RL policy code unchanged:
    IsaacLab env obs -> old ActorCritic.act_inference(obs, hist_encoding=True) -> IsaacLab env.step(actions)
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import torch
import time

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LOW_LEVEL_ROOT.parent
RSL_RL_ROOT = REPO_ROOT / "third_party" / "rsl_rl"

# Make sure current low-level code and old rsl_rl are imported first.
for p in [str(LOW_LEVEL_ROOT), str(RSL_RL_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

print("[play_isaaclab_policy] LOW_LEVEL_ROOT:", LOW_LEVEL_ROOT)
print("[play_isaaclab_policy] RSL_RL_ROOT:", RSL_RL_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot_urdf_path",
    type=str,
    default="/workspace/visual_wholebody_doublemode/low-level/resources/robots/b2z1/urdf/b2z1_isaacsim_mesh_axis_fixed.urdf",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/workspace/visual_wholebody_doublemode/low-level/ckpt/调权重_robotlab_等高系_2/model_25000.pt",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--print_every", type=int, default=1)

parser.add_argument("--debug_dump_path", type=str, default="")
parser.add_argument("--debug_steps", type=int, default=0)
parser.add_argument("--cmd_vx", type=float, default=0.0)
parser.add_argument("--cmd_yaw", type=float, default=0.0)
parser.add_argument("--action_clip", type=float, default=0.0)  # 0 means no clamp

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from legged_gym.envs.manip_loco.b2z1_isaaclab_config import B2Z1IsaacLabCfg
from legged_gym.envs.manip_loco.manip_loco_isaaclab import ManipLocoIsaacLab
from rsl_rl.modules.actor_critic import ActorCritic


def unwrap_obs(obs_out):
    """IsaacLab may return either tensor obs or {'policy': tensor}."""
    if isinstance(obs_out, dict):
        return obs_out["policy"]
    return obs_out


def load_policy(ckpt_path: str, device: torch.device | str):
    """Load old visual_wholebody / rsl_rl ActorCritic checkpoint.

    The checkpoint was trained with:
      num_prop=72, num_priv=18, history_len=10,
      leg actions=12, arm actions=6, total actions=18.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    std_init = ckpt["model_state_dict"]["std"].detach().cpu().tolist()
    actor_critic = ActorCritic(
        num_actor_obs=72,
        num_critic_obs=72,
        num_actions=18,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        leg_control_head_hidden_dims=[128, 128],
        arm_control_head_hidden_dims=[128, 128],
        priv_encoder_dims=[64, 20],
        activation="elu",
        # init_std=1.0,
        init_std=std_init,
        num_leg_actions=12,
        num_arm_actions=6,
        adaptive_arm_gains=False,
        adaptive_arm_gains_scale=1.0,
        num_priv=18,
        num_hist=10,
        num_prop=72,
        output_tanh=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    actor_critic.eval()

    print("[policy] loaded checkpoint:", ckpt_path)
    print("[policy] iter:", ckpt.get("iter", None))
    print("[policy] next_learning_iteration:", ckpt.get("next_learning_iteration", None))

    return actor_critic.act_inference

def cpu(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def collect_debug_frame(env, obs, actions_raw, actions_to_env, step):
    num_prop = int(getattr(env.cfg, "num_proprio", 72))
    num_priv = int(getattr(env.cfg, "num_priv", 18))
    history_len = int(getattr(env.cfg, "history_len", 10))

    frame = {
        "step": step,
        "dt": float(env.step_dt),
        "obs": cpu(obs),
        "obs_prop": cpu(obs[:, :num_prop]),
        "obs_priv": cpu(obs[:, num_prop:num_prop + num_priv]),
        "obs_hist": cpu(obs[:, -history_len * num_prop:]),
        "actions_raw": cpu(actions_raw),
        "actions_to_env": cpu(actions_to_env),
        "commands": cpu(env.commands),
        "base_lin_vel": cpu(env.base_lin_vel),
        "base_ang_vel": cpu(env.base_ang_vel),
        "root_states": cpu(env.root_states),
        "dof_pos": cpu(env.dof_pos),
        "dof_vel": cpu(env.dof_vel),
        "dof_pos_policy_all": cpu(env._env_to_policy_all(env.dof_pos)),
        "dof_vel_policy_all": cpu(env._env_to_policy_all(env.dof_vel)),
        "default_dof_pos_policy_all": cpu(env._env_to_policy_all(env.default_dof_pos)),
        "action_history_last": cpu(env.action_history_buf[:, -1]),
        "gait_indices": cpu(getattr(env, "gait_indices", None)),
        "clock_inputs": cpu(getattr(env, "clock_inputs", None)),
        "foot_contacts": cpu(getattr(env, "foot_contacts_from_sensor", None)),
        "curr_ee_goal_cart": cpu(getattr(env, "curr_ee_goal_cart", None)),
        "curr_ee_goal_sphere": cpu(getattr(env, "curr_ee_goal_sphere", None)),
        "goal_height_follow_mask": cpu(getattr(env, "goal_height_follow_mask", None)),
        "target_pos": cpu(getattr(env, "target_pos", None)),
    }
    return frame

cfg = B2Z1IsaacLabCfg()
cfg.robot_urdf_path = args.robot_urdf_path
cfg.robot.spawn.asset_path = args.robot_urdf_path
cfg.scene.num_envs = args.num_envs

env = ManipLocoIsaacLab(cfg)
obs_out, _ = env.reset()
env.commands[:] = 0.0
env.commands[:, 0] = args.cmd_vx
env.commands[:, 2] = args.cmd_yaw
# obs = unwrap_obs(obs_out)
obs = unwrap_obs(env.get_observations())

print("joint_names:", env.robot.data.joint_names)
print("body_names:", env.robot.data.body_names)
print("[obs] shape:", obs.shape)

assert obs.shape[-1] == 810, f"Policy expects obs dim 810, got {obs.shape}"

policy = load_policy(args.ckpt_path, env.device)

print("[torch] cuda available:", torch.cuda.is_available())
print("[torch] current device:", torch.cuda.current_device())
print("[torch] device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

try:
    print("[policy] first param device:", next(policy.parameters()).device)
except Exception as e:
    print("[policy] cannot read param device:", e)

print("[obs] device:", obs.device)

debug_frames = []

step = 0
print("env.step_dt is: ", env.step_dt)
while simulation_app.is_running():
    with torch.inference_mode():
        env.commands[:] = 0.0
        env.commands[:, 0] = args.cmd_vx
        env.commands[:, 2] = args.cmd_yaw

        # Important: recompute obs after setting command, like old debug path.
        obs_out = env.get_observations()
        obs = unwrap_obs(obs_out)

        actions_raw = policy(obs.detach(), hist_encoding=True)

        if args.action_clip > 0:
            actions_to_env = torch.clamp(actions_raw, -args.action_clip, args.action_clip)
        else:
            actions_to_env = actions_raw

        if args.debug_dump_path:
            debug_frames.append(collect_debug_frame(env, obs, actions_raw, actions_to_env, step))

        obs_out, rew, terminated, truncated, info = env.step(actions_to_env.detach())
        # obs = unwrap_obs(obs_out)

        if step % args.print_every == 0:
            print(
                f"[new step={step}] "
                f"cmd={env.commands[0, :3].detach().cpu().tolist()} "
                f"base_lin_vel={env.base_lin_vel[0].detach().cpu().tolist()} "
                f"raw_leg=({float(actions_raw[:, :12].min()):+.3f},{float(actions_raw[:, :12].max()):+.3f}) "
                f"raw_arm=({float(actions_raw[:, 12:].min()):+.3f},{float(actions_raw[:, 12:].max()):+.3f}) "
                f"to_env_leg=({float(actions_to_env[:, :12].min()):+.3f},{float(actions_to_env[:, :12].max()):+.3f})"
            )

        step += 1

        if args.debug_dump_path and step >= args.debug_steps:
            torch.save(
                {
                    "source": "new_isaaclab",
                    "frames": debug_frames,
                },
                args.debug_dump_path,
            )
            print(f"[DEBUG] saved new dump to {args.debug_dump_path}")
            break

# step = 0
# wall_t0 = time.perf_counter()
# last_print_wall_t = wall_t0

# dt = env.step_dt
# print("env.step_dt is:", env.step_dt)
# print("expected control freq is:", 1.0 / env.step_dt, "Hz")

# while simulation_app.is_running():
#     time_start = time.time()
#     loop_start_t = time.perf_counter()

#     with torch.inference_mode():
#         env.commands[:] = 0.0
#         env.commands[:, 0] = args.cmd_vx
#         env.commands[:, 2] = args.cmd_yaw

#         # -------------------------
#         # 1. policy inference timing
#         # -------------------------
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         policy_t0 = time.perf_counter()

#         actions_raw = policy(obs.detach(), hist_encoding=True)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         policy_t1 = time.perf_counter()
#         policy_time_ms = (policy_t1 - policy_t0) * 1000.0

#         # -------------------------
#         # action clipping
#         # -------------------------
#         if args.action_clip > 0:
#             actions_to_env = torch.clamp(actions_raw, -args.action_clip, args.action_clip)
#         else:
#             actions_to_env = actions_raw

#         if args.debug_dump_path:
#             debug_frames.append(
#                 collect_debug_frame(env, obs, actions_raw, actions_to_env, step)
#             )

#         # -------------------------
#         # 2. env.step timing
#         # -------------------------
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         env_step_t0 = time.perf_counter()

#         obs_out, rew, terminated, truncated, info = env.step(actions_to_env.detach())

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         env_step_t1 = time.perf_counter()
#         env_step_time_ms = (env_step_t1 - env_step_t0) * 1000.0

#         obs = unwrap_obs(obs_out)

#         # -------------------------
#         # 3. loop timing
#         # -------------------------
#         loop_end_t = time.perf_counter()
#         loop_time_ms = (loop_end_t - loop_start_t) * 1000.0

#         # expected sim time from step counter
#         sim_time_from_step = step * env.step_dt

#         # real elapsed wall time
#         wall_elapsed = loop_end_t - wall_t0

#         # real-time factor
#         if wall_elapsed > 0:
#             rtf = sim_time_from_step / wall_elapsed
#         else:
#             rtf = 0.0

#         # actual loop frequency
#         if loop_time_ms > 0:
#             instant_hz = 1000.0 / loop_time_ms
#         else:
#             instant_hz = 0.0

#         if step % args.print_every == 0:
#             now = time.perf_counter()
#             interval_wall = now - last_print_wall_t

#             if interval_wall > 0:
#                 avg_hz_since_last_print = args.print_every / interval_wall
#             else:
#                 avg_hz_since_last_print = 0.0

#             last_print_wall_t = now

#             # print(
#             #     f"[new step={step}] "
#             #     f"sim_time={sim_time_from_step:.3f}s "
#             #     f"wall_elapsed={wall_elapsed:.3f}s "
#             #     f"rtf={rtf:.3f} "
#             #     f"instant_hz={instant_hz:.2f} "
#             #     f"avg_hz={avg_hz_since_last_print:.2f} "
#             #     f"policy={policy_time_ms:.3f}ms "
#             #     f"env_step={env_step_time_ms:.3f}ms "
#             #     f"loop={loop_time_ms:.3f}ms "
#             #     f"cmd={env.commands[0, :3].detach().cpu().tolist()} "
#             #     f"base_lin_vel={env.base_lin_vel[0].detach().cpu().tolist()} "
#             #     f"raw_leg=({float(actions_raw[:, :12].min()):+.3f},{float(actions_raw[:, :12].max()):+.3f}) "
#             #     f"raw_arm=({float(actions_raw[:, 12:].min()):+.3f},{float(actions_raw[:, 12:].max()):+.3f}) "
#             #     f"to_env_leg=({float(actions_to_env[:, :12].min()):+.3f},{float(actions_to_env[:, :12].max()):+.3f})"
#             # )

#         step += 1

#         if args.debug_dump_path and step >= args.debug_steps:
#             torch.save(
#                 {
#                     "source": "new_isaaclab",
#                     "frames": debug_frames,
#                 },
#                 args.debug_dump_path,
#             )
#             print(f"[DEBUG] saved new dump to {args.debug_dump_path}")
#             break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - time_start)
#         if sleep_time > 0:
#             print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
#             time.sleep(sleep_time)        

env.close()
simulation_app.close()
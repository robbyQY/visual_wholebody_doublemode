from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch

LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LOW_LEVEL_ROOT.parent
RSL_RL_ROOT = REPO_ROOT / "third_party" / "rsl_rl"
for p in [str(LOW_LEVEL_ROOT), str(RSL_RL_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot_urdf_path", type=str, default="", help="Explicit URDF path. If empty, auto-generate from mount/ablation args.")
parser.add_argument("--base_robot", type=str, default="b2z1", choices=["b1z1", "b2z1"])
parser.add_argument("--run_metadata_path", type=str, default="", help="Path to run_metadata.json (auto-detected from ckpt dir if empty).")
parser.add_argument("--mount_deg", type=float, default=0.0)
parser.add_argument("--mount_xyz", type=float, nargs=3, default=None)
parser.add_argument("--robot_ablation", type=str, default="")
parser.add_argument("--leg_collision_scale", type=float, default=1.0)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--teleop_mode", action="store_true")
parser.add_argument("--stdin_teleop", action="store_true", help="(Optional) Read keyboard from terminal stdin (raw mode).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# IMPORTANT: import Isaac Sim / Kit modules only after AppLauncher starts the app.
import carb
import carb.input
import omni.appwindow

from legged_gym.envs.manip_loco.b2z1_config import B2Z1IsaacLabCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLocoIsaacLab
from rsl_rl.modules.actor_critic import ActorCritic

from legged_gym.utils.b1z1_mount import ensure_mount_urdf, MOUNT_URDF_SPECS
from legged_gym.utils.robot_ablation import ensure_cross_robot_ablation_urdf

def _load_checkpoint_features(args) -> dict:
    metadata_path = Path(args.run_metadata_path) if args.run_metadata_path else Path(args.ckpt_path).resolve().parent / "run_metadata.json"
    if not metadata_path.exists():
        print(f"[urdf] run metadata not found: {metadata_path}; use CLI args/defaults.")
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[urdf] failed to parse run metadata {metadata_path}: {exc}; use CLI args/defaults.")
        return {}
    features = metadata.get("checkpoint_features", {})
    print(f"[urdf] loaded checkpoint_features from: {metadata_path}")
    return features if isinstance(features, dict) else {}


def resolve_robot_urdf_path(args) -> str:
    if args.robot_urdf_path:
        return args.robot_urdf_path

    root_dir = str(LOW_LEVEL_ROOT)
    features = _load_checkpoint_features(args)

    mount_deg = float(features.get("mount_deg", args.mount_deg))
    mount_xyz = args.mount_xyz
    if mount_xyz is None:
        if all(k in features for k in ("mount_x", "mount_y", "mount_z")):
            mount_xyz = [features["mount_x"], features["mount_y"], features["mount_z"]]
        else:
            mount_xyz = MOUNT_URDF_SPECS[args.base_robot]["default_xyz"]

    robot_ablation = args.robot_ablation.strip()
    if not robot_ablation and "robot_ablation" in features:
        robot_ablation = str(features.get("robot_ablation") or "")
    robot_ablation = robot_ablation.strip().lower()
    # Match legacy semantics: "none" means no ablation.
    if robot_ablation in ("", "none"):
        robot_ablation = None

    leg_collision_scale = float(features.get("leg_collision_scale", args.leg_collision_scale))

    need_ablation = robot_ablation is not None or leg_collision_scale != 1.0
    if need_ablation:
        urdf_rel = ensure_cross_robot_ablation_urdf(
            root_dir=root_dir,
            base_robot=args.base_robot,
            robot_ablation=robot_ablation,
            mount_deg=mount_deg,
            mount_xyz=mount_xyz,
            leg_collision_scale=leg_collision_scale,
        )
    else:
        urdf_rel = ensure_mount_urdf(
            root_dir=root_dir,
            generator_name=args.base_robot,
            mount_deg=mount_deg,
            mount_xyz=mount_xyz,
        )

    urdf_path = str((LOW_LEVEL_ROOT / urdf_rel).resolve())
    print(f"[urdf] auto-generated: {urdf_path}")
    return urdf_path

# Isaac Sim native keyboard subscription (viewer-focused input).
_keyboard_sub = None

cfg = B2Z1IsaacLabCfg()
resolved_urdf_path = resolve_robot_urdf_path(args)
cfg.robot_urdf_path = resolved_urdf_path
cfg.robot.spawn.asset_path = resolved_urdf_path
cfg.scene.num_envs = args.num_envs
cfg.env.teleop_mode = args.teleop_mode
cfg.sim.render_interval = 4

env = ManipLocoIsaacLab(cfg)
obs_out, _ = env.reset()
obs = obs_out["policy"] if isinstance(obs_out, dict) else obs_out

def load_policy(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    std_init = ckpt["model_state_dict"]["std"].detach().cpu().tolist()
    actor_critic = ActorCritic(
        num_actor_obs=72, num_critic_obs=72, num_actions=18,
        actor_hidden_dims=[128], critic_hidden_dims=[128],
        leg_control_head_hidden_dims=[128, 128], arm_control_head_hidden_dims=[128, 128],
        priv_encoder_dims=[64, 20], activation="elu", init_std=std_init,
        num_leg_actions=12, num_arm_actions=6, adaptive_arm_gains=False,
        adaptive_arm_gains_scale=1.0, num_priv=18, num_hist=10, num_prop=72, output_tanh=False,
    ).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    actor_critic.eval()
    return actor_critic.act_inference

policy = load_policy(args.ckpt_path, env.device)

_TELEOP_KEYS = tuple("qweasdg yvujikzxcmbnlop".replace(" ", ""))


def _event_type_is_press(event_type) -> bool:
    """Return True for key press events across Isaac Sim / carb.input versions."""
    press_types = []
    keyboard_event_type = getattr(carb.input, "KeyboardEventType", None)
    if keyboard_event_type is not None:
        for name in ("KEY_PRESS", "KEY_REPEAT"):
            value = getattr(keyboard_event_type, name, None)
            if value is not None:
                press_types.append(value)

    for value in press_types:
        try:
            if int(event_type) == int(value):
                return True
        except Exception:
            if event_type == value:
                return True

    # Conservative fallback: many versions use 1 for KEY_PRESS.
    try:
        return int(event_type) == 1
    except Exception:
        return False


def _keyboard_input_to_key(input_value) -> str | None:
    """Convert carb.input KeyboardInput enum/int/string to the legacy teleop key.

    Some Isaac Sim builds do not expose carb.input.get_keyboard_key_name(), so
    map the enum manually. This supports both KeyboardInput.W and KEY_W naming.
    """
    keyboard_input = getattr(carb.input, "KeyboardInput", None)

    if keyboard_input is not None:
        for key in _TELEOP_KEYS:
            upper = key.upper()
            for attr in (upper, f"KEY_{upper}"):
                enum_value = getattr(keyboard_input, attr, None)
                if enum_value is None:
                    continue
                try:
                    if int(input_value) == int(enum_value):
                        return key
                except Exception:
                    if input_value == enum_value:
                        return key

    # Fallback for enum strings like "KeyboardInput.W", "KEY_W", or "87".
    s = str(input_value).strip().lower()
    s = s.replace("keyboardinput.", "").replace("key_", "")
    if len(s) == 1 and s in _TELEOP_KEYS:
        return s
    for key in _TELEOP_KEYS:
        if s.endswith(f".{key}") or s.endswith(f"_{key}") or s == key:
            return key

    return None


def _install_isaacsim_keyboard():
    global _keyboard_sub
    app_window = omni.appwindow.get_default_app_window()
    if app_window is None:
        print("[teleop][warn] no default app window; IsaacSim keyboard input unavailable.")
        return
    kb = app_window.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    if kb is None or input_iface is None:
        print("[teleop][warn] keyboard/input interface unavailable.")
        return

    def on_key_event(event, *args, **kwargs):
        if not _event_type_is_press(event.type):
            return True

        key = _keyboard_input_to_key(event.input)
        if key is None:
            # Do not throw inside the Kit callback; just ignore unknown keys.
            return True

        env.apply_teleop_key(key)
        return True

    _keyboard_sub = input_iface.subscribe_to_keyboard_events(kb, on_key_event)
    print("[teleop] IsaacSim keyboard subscribed. Click viewport, then use WASD/YUHJ.../OP/G.")


if args.teleop_mode:
    _install_isaacsim_keyboard()

import time
step = 0
wall_t0 = time.perf_counter()
last_wall = wall_t0
last_sim = 0.0

while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs.detach(), hist_encoding=True)
        obs_out, *_ = env.step(actions.detach())
        obs = obs_out["policy"] if isinstance(obs_out, dict) else obs_out

        sim_time = (step + 1) * env.dt

        if step % 50 == 0 and step > 0:
            now = time.perf_counter()
            wall_dt = now - last_wall
            sim_dt = sim_time - last_sim
            rtf = sim_dt / wall_dt if wall_dt > 0 else 0.0

            vel = env.robot.data.root_lin_vel_w[0].detach().cpu().tolist()
            print(
                f"[sim={sim_time:.2f}s wall_dt={wall_dt:.2f}s sim_dt={sim_dt:.2f}s RTF={rtf:.3f}] "
                f"step={step} cmd={env.commands[0,:3].detach().cpu().tolist()} vel={vel} "
                f"arm_mode={env.teleop_arm_control_mode} ee_goal={env.curr_ee_goal_cart[0].detach().cpu().tolist()}"
            )
            last_wall = now
            last_sim = sim_time

        step += 1

if _keyboard_sub is not None:
    try:
        app_window = omni.appwindow.get_default_app_window()
        if app_window is not None:
            kb = app_window.get_keyboard()
            carb.input.acquire_input_interface().unsubscribe_from_keyboard_events(kb, _keyboard_sub)
    except Exception:
        pass
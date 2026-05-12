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

import numpy as np
import os
from datetime import datetime
import glob
import sys
import signal
import isaacgym
import torch.distributed as dist

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
from legged_gym.utils.helpers import load_run_metadata, save_run_metadata, update_run_metadata, get_run_log_dir, update_cfg_from_args, _extract_checkpoint_features, apply_checkpoint_features_from_run
import torch
import wandb

def _make_training_signal_handler(signal_name):
    def _handler(signum, frame):
        raise KeyboardInterrupt(f"Received {signal_name} ({signum})")
    return _handler


def install_training_signal_handlers():
    signal_names = {
        signal.SIGINT: "SIGINT",
        signal.SIGTERM: "SIGTERM",
    }
    previous_handlers = {}
    for sig, sig_name in signal_names.items():
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _make_training_signal_handler(sig_name))
    return previous_handlers


def restore_training_signal_handlers(previous_handlers):
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def safe_finish_wandb(wandb_initialized, cancelled):
    if not wandb_initialized:
        return
    try:
        wandb.finish(exit_code=0 if cancelled else None)
    except Exception as exc:
        print(f"Warning: wandb.finish failed: {exc}", file=sys.stderr)


def safe_cleanup_distributed(cancelled):
    if not (dist.is_available() and dist.is_initialized()):
        return

    if not cancelled:
        try:
            dist.barrier()
        except Exception as exc:
            print(f"Warning: dist.barrier failed: {exc}", file=sys.stderr)

    try:
        dist.destroy_process_group()
    except Exception as exc:
        print(f"Warning: dist.destroy_process_group failed: {exc}", file=sys.stderr)


def _wandb_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _wandb_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_wandb_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def build_wandb_config(args, env_cfg, train_cfg, log_pth):
    return _wandb_safe({
        "cli_args": vars(args),
        "env_cfg": class_to_dict(env_cfg),
        "train_cfg": class_to_dict(train_cfg),
        "runtime": {
            "log_dir": log_pth,
            "distributed": args.distributed,
            "world_size": args.world_size,
            "rank": args.rank,
            "local_rank": args.local_rank,
            "sim_device": args.sim_device,
            "rl_device": args.rl_device,
        },
    })

def get_wandb_init_kwargs(args, log_pth, mode):
    init_kwargs = {
        "project": args.proj_name,
        "name": args.exptid,
        "mode": mode,
        "dir": LEGGED_GYM_ENVS_DIR + "/logs",
    }
    if getattr(args, "wandb_group", ""):
        init_kwargs["group"] = args.wandb_group
    if mode == "disabled":
        return init_kwargs

    if args.resume:
        metadata = load_run_metadata(log_pth, filename=getattr(args, "run_metadata_filename", None)) or {}
        wandb_info = metadata.get("wandb", {})
        run_id = wandb_info.get("run_id")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "must"
        else:
            print(f"Warning: no wandb run_id found under {log_pth}; resume mode will start a new wandb run.")
    return init_kwargs

class TimestampedTee:
    def __init__(self, stream, log_path, rank):
        self.stream = stream
        self.rank = rank
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0
        self.stream.write(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log.write(f"[{timestamp}] [rank {self.rank}] {line}\n")
        self.log.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        if self._buffer:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log.write(f"[{timestamp}] [rank {self.rank}] {self._buffer}\n")
            self._buffer = ""
        self.log.flush()

def _next_available_exptid(proj_name, base_exptid):
    suffix = 2
    candidate = base_exptid
    while os.path.exists(get_run_log_dir(proj_name, candidate)):
        candidate = f"{base_exptid}_{suffix}"
        suffix += 1
    return candidate

def _next_available_file(log_dir, stem, extension):
    suffix = 2
    while True:
        candidate = os.path.join(log_dir, f"{stem}_{suffix}{extension}")
        if not os.path.exists(candidate):
            return candidate, suffix
        suffix += 1

def _has_any_checkpoint(log_dir):
    return bool(glob.glob(os.path.join(log_dir, "model_*.pt")))

def resolve_training_context(args):
    requested_exptid = args.exptid
    train_mode = getattr(args, "train_mode", "fresh")
    args.requested_exptid = requested_exptid
    args.run_metadata_filename = "run_metadata.json"
    args.log_file_path = None
    args.resume = False
    args.resumeid = None

    if train_mode == "resume":
        log_dir = get_run_log_dir(args.proj_name, requested_exptid)
        if _has_any_checkpoint(log_dir):
            args.resume = True
            args.resumeid = requested_exptid
            args.exptid = requested_exptid
            args.effective_train_mode = "resume"
            args.log_file_path = os.path.join(log_dir, "train.log")
            return args, log_dir

        os.makedirs(log_dir, exist_ok=True)
        log_file_path, suffix = _next_available_file(log_dir, "train", ".log")
        args.run_metadata_filename = f"run_metadata_{suffix}.json"
        args.exptid = requested_exptid
        args.effective_train_mode = "fresh"
        args.log_file_path = log_file_path
        return args, log_dir

    if train_mode in ("fresh", "load"):
        args.exptid = _next_available_exptid(args.proj_name, requested_exptid) if os.path.exists(get_run_log_dir(args.proj_name, requested_exptid)) else requested_exptid
        args.effective_train_mode = train_mode
        log_dir = get_run_log_dir(args.proj_name, args.exptid)
        args.log_file_path = os.path.join(log_dir, "train.log")
        return args, log_dir

    raise ValueError(f"Unsupported train_mode={train_mode}")

def resolve_training_context_distributed(args):
    if not getattr(args, "distributed", False) or not dist.is_initialized():
        return resolve_training_context(args)

    payload = [None]
    if args.rank == 0:
        args, log_dir = resolve_training_context(args)
        os.makedirs(log_dir, exist_ok=True)
        payload[0] = {
            "log_dir": log_dir,
            "requested_exptid": getattr(args, "requested_exptid", None),
            "run_metadata_filename": getattr(args, "run_metadata_filename", "run_metadata.json"),
            "log_file_path": getattr(args, "log_file_path", None),
            "resume": getattr(args, "resume", False),
            "resumeid": getattr(args, "resumeid", None),
            "exptid": getattr(args, "exptid", None),
            "effective_train_mode": getattr(args, "effective_train_mode", getattr(args, "train_mode", "fresh")),
        }

    dist.broadcast_object_list(payload, src=0)
    context = payload[0]

    args.requested_exptid = context["requested_exptid"]
    args.run_metadata_filename = context["run_metadata_filename"]
    args.log_file_path = context["log_file_path"]
    args.resume = context["resume"]
    args.resumeid = context["resumeid"]
    args.exptid = context["exptid"]
    args.effective_train_mode = context["effective_train_mode"]
    return args, context["log_dir"]

def log_training_header(args, log_pth, num_gpus, env_cfg=None):
    if args.rank != 0:
        return
    distributed = num_gpus > 1
    total_num_envs = args.num_envs if args.num_envs is not None else "<config>"
    if total_num_envs != "<config>" and num_gpus > 0:
        num_envs_per_gpu = total_num_envs // num_gpus if total_num_envs % num_gpus == 0 else "<invalid: not divisible>"
    else:
        num_envs_per_gpu = "<resolved in Python>"

    print("==== Training Config ====")
    print(f"ROOT_DIR={LEGGED_GYM_ROOT_DIR}")
    print(f"PROJ_NAME={args.proj_name}")
    print(f"REQUESTED_EXPTID={getattr(args, 'requested_exptid', args.exptid)}")
    print(f"EXPTID={args.exptid}")
    print(f"TASK={args.task}")
    print(f"LOG_ROOT={os.path.dirname(os.path.dirname(log_pth))}")
    print(f"LOG_DIR={log_pth}")
    print(f"LOG_FILE={getattr(args, 'log_file_path', '<none>')}")
    print(f"RUN_METADATA_FILENAME={getattr(args, 'run_metadata_filename', 'run_metadata.json')}")
    print(f"TRAIN_MODE={getattr(args, 'train_mode', 'fresh')}")
    print(f"EFFECTIVE_TRAIN_MODE={getattr(args, 'effective_train_mode', getattr(args, 'train_mode', 'fresh'))}")
    print(f"LOAD_EXPTID={getattr(args, 'load_exptid', None) or '<none>'}")
    print(f"LOAD_CKPT={getattr(args, 'checkpoint', '<default>')}")
    print(f"TRAIN_LOG_EVERY={getattr(args, 'train_log_every', 1)}")
    print(f"WANDB_GROUP={getattr(args, 'wandb_group', '') or '<none>'}")
    print(f"NUM_ENVS={total_num_envs}")
    print(f"NUM_GPUS={num_gpus}")
    print(f"DISTRIBUTED={distributed}")
    print(f"NUM_ENVS_PER_GPU={num_envs_per_gpu}")
    print(f"OBSERVE_GAIT_COMMANDS={env_cfg.env.observe_gait_commands}")
    print(f"MIXED_HEIGHT_REFERENCE={env_cfg.goal_ee.sphere_center.mixed_height_reference}")
    print(f"TRUNK_FOLLOW_RATIO={env_cfg.goal_ee.sphere_center.trunk_follow_ratio}")
    print(f"OMNIDIRECTIONAL_POS_Y={env_cfg.goal_ee.ranges.omnidirectional_pos_y}")
    checkpoint_features = _extract_checkpoint_features(args, env_cfg)
    print(f"ROBOT_ABLATION={checkpoint_features['robot_ablation']}")
    print(f"LEG_COLLISION_SCALE={checkpoint_features['leg_collision_scale']}")
    print(f"MOUNT_DEG={checkpoint_features['mount_deg'] if 'mount_deg' in checkpoint_features else None}")
    print(f"MOUNT_X={checkpoint_features['mount_x'] if 'mount_x' in checkpoint_features else None}")
    print(f"MOUNT_Y={checkpoint_features['mount_y'] if 'mount_y' in checkpoint_features else None}")
    print(f"MOUNT_Z={checkpoint_features['mount_z'] if 'mount_z' in checkpoint_features else None}")
    if not env_cfg.goal_ee.ranges.omnidirectional_pos_y:
        print(f"NON_OMNI_POS_Y_SCHEDULE={env_cfg.commands.non_omni_pos_y_schedule}")
    print(f"EE_GOAL_OBS_MODE={env_cfg.env.ee_goal_obs_mode}")
    print(f"REWARD_SCALE_PRESET={env_cfg.rewards.reward_scale_preset}")
    print(f"GAIT_FREQUENCY_MIN={env_cfg.env.gait_frequency_min}")
    print(f"GAIT_FREQUENCY_MAX={env_cfg.env.gait_frequency_max}")
    print(f"GAIT_FREQUENCY_LIN_VEL_REF={env_cfg.env.gait_frequency_lin_vel_ref}")
    print(f"GAIT_FREQUENCY_ANG_VEL_REF={env_cfg.env.gait_frequency_ang_vel_ref}")
    print(f"GAIT_FREQUENCY_ANG_VEL_WEIGHT={env_cfg.env.gait_frequency_ang_vel_weight}")
    print(f"START_TIME={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    args.distributed = distributed
    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA, but CUDA is not available")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.rl_device = f"cuda:{local_rank}"
        args.sim_device_id = local_rank
        args.sim_device = f"cuda:{local_rank}"
    return args

def train(args):
    args = setup_distributed(args)
    previous_signal_handlers = install_training_signal_handlers()
    wandb_initialized = False
    cancelled = False
    args, log_pth = resolve_training_context_distributed(args)
    os.makedirs(log_pth, exist_ok=True)
    try:
        if args.rank == 0 and args.log_file_path:
            sys.stdout = TimestampedTee(sys.stdout, args.log_file_path, args.rank)
            sys.stderr = TimestampedTee(sys.stderr, args.log_file_path, args.rank)
        if getattr(args, "effective_train_mode", getattr(args, "train_mode", "fresh")) == "resume":
            args, _ = apply_checkpoint_features_from_run(
                args,
                log_pth,
                verbose=args.rank == 0,
                filename=getattr(args, "run_metadata_filename", None),
            )
        env_cfg_preview, _ = task_registry.get_cfgs(args.task)
        env_cfg_preview, _ = update_cfg_from_args(env_cfg_preview, None, args)
        log_training_header(args, log_pth, args.world_size if args.distributed else 1, env_cfg_preview)
        if args.debug:
            mode = "disabled"
            args.rows = 6
            args.cols = 2
            args.num_envs = 128
        else:
            mode = "online"
        if args.rank != 0:
            mode = "disabled"
        wandb.init(**get_wandb_init_kwargs(args, log_pth, mode))
        wandb_initialized = True

        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg_preview)
        ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)

        if args.rank == 0:
            save_run_metadata(log_pth, args, env_cfg, train_cfg, filename=args.run_metadata_filename)
            update_run_metadata(log_pth, {
                "wandb": {
                    "entity": getattr(wandb.run, "entity", None),
                    "name": getattr(wandb.run, "name", None),
                    "project": getattr(wandb.run, "project", None),
                    "run_id": getattr(wandb.run, "id", None),
                    "url": getattr(wandb.run, "url", None),
                }
            }, filename=args.run_metadata_filename)
            wandb.config.update(build_wandb_config(args, env_cfg, train_cfg, log_pth), allow_val_change=True)
            task_config_path = os.path.join(LEGGED_GYM_ENVS_DIR, "manip_loco", f"{args.task}_config.py")
            if os.path.isfile(task_config_path):
                wandb.save(task_config_path, policy="now")
            base_config_path = os.path.join(LEGGED_GYM_ENVS_DIR, "manip_loco", "manip_loco_base_config.py")
            if os.path.isfile(base_config_path):
                wandb.save(base_config_path, policy="now")
            wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt as exc:
        cancelled = True
        print(f"Training cancelled: {exc}", file=sys.stderr)
    finally:
        safe_finish_wandb(wandb_initialized, cancelled)
        safe_cleanup_distributed(cancelled)
        restore_training_signal_handlers(previous_signal_handlers)

if __name__ == '__main__':
    args = get_args()
    train(args)

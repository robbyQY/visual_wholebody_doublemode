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

import os
import copy
import sys
import json
import re
import glob
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

RUN_METADATA_FILENAME = "run_metadata.json"

def get_log_root():
    return os.environ.get("LEGGED_GYM_LOG_ROOT", os.path.join(LEGGED_GYM_ROOT_DIR, "logs"))

def get_run_log_dir(proj_name, exptid):
    return os.path.join(get_log_root(), proj_name, exptid)

def get_run_metadata_filename(filename=None):
    if filename is not None:
        return filename
    return os.environ.get("LEGGED_GYM_RUN_METADATA_FILENAME", RUN_METADATA_FILENAME)

def get_run_metadata_path(log_dir, filename=None):
    return os.path.join(log_dir, get_run_metadata_filename(filename))

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def _parse_schedule_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return None
        return [float(part) for part in parts]
    return [float(value)]

def _extract_checkpoint_features(args, env_cfg):
    observe_gait_commands = False
    mixed_height_reference = False
    omnidirectional_pos_y = False
    if env_cfg is not None:
        observe_gait_commands = bool(getattr(env_cfg.env, "observe_gait_commands", False))
        mixed_height_reference = bool(getattr(env_cfg.goal_ee.sphere_center, "mixed_height_reference", False))
        omnidirectional_pos_y = bool(getattr(env_cfg.goal_ee.ranges, "omnidirectional_pos_y", False))
    else:
        observe_gait_commands = bool(getattr(args, "observe_gait_commands", False))
        mixed_height_reference = bool(getattr(args, "mixed_height_reference", False))
        omnidirectional_pos_y = bool(getattr(args, "omnidirectional_pos_y", False))
    return {
        "observe_gait_commands": observe_gait_commands,
        "mixed_height_reference": mixed_height_reference,
        "omnidirectional_pos_y": omnidirectional_pos_y,
    }

def save_run_metadata(log_dir, args, env_cfg=None, train_cfg=None, filename=None):
    os.makedirs(log_dir, exist_ok=True)
    metadata = {
        "checkpoint_features": _extract_checkpoint_features(args, env_cfg),
        "cli_args": _json_safe(vars(args)),
    }
    if env_cfg is not None:
        metadata["env_cfg"] = _json_safe(class_to_dict(env_cfg))
    if train_cfg is not None:
        metadata["train_cfg"] = _json_safe(class_to_dict(train_cfg))

    metadata_path = get_run_metadata_path(log_dir, filename=filename)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path

def update_run_metadata(log_dir, updates, filename=None):
    os.makedirs(log_dir, exist_ok=True)
    metadata = load_run_metadata(log_dir, filename=filename) or {}
    metadata.pop("_source", None)
    for key, value in updates.items():
        metadata[key] = _json_safe(value)

    metadata_path = get_run_metadata_path(log_dir, filename=filename)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path

def load_matching_checkpoint_weights(module, checkpoint_path, device="cpu", verbose=True):
    loaded_obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
        checkpoint_state = loaded_obj["model_state_dict"]
    elif isinstance(loaded_obj, dict):
        checkpoint_state = loaded_obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(loaded_obj)}")

    model_state = module.state_dict()
    matched_state = {}
    missing_keys = []
    unexpected_keys = []
    shape_mismatched_keys = []

    for key, tensor in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != tensor.shape:
            shape_mismatched_keys.append(
                f"{key} (checkpoint={tuple(tensor.shape)}, model={tuple(model_state[key].shape)})"
            )
            continue
        matched_state[key] = tensor

    for key in model_state.keys():
        if key not in checkpoint_state:
            missing_keys.append(key)

    load_result = module.load_state_dict(matched_state, strict=False)

    if verbose:
        print(f"Partially loaded checkpoint weights from: {checkpoint_path}")
        print(
            "Checkpoint load summary: matched={}, missing={}, unexpected={}, shape_mismatched={}".format(
                len(matched_state),
                len(missing_keys),
                len(unexpected_keys),
                len(shape_mismatched_keys),
            )
        )
        if missing_keys:
            print("Warning: missing keys not loaded:")
            for key in missing_keys:
                print(f"  - {key}")
        if unexpected_keys:
            print("Warning: unexpected checkpoint keys skipped:")
            for key in unexpected_keys:
                print(f"  - {key}")
        if shape_mismatched_keys:
            print("Warning: shape-mismatched keys skipped:")
            for key in shape_mismatched_keys:
                print(f"  - {key}")
        if getattr(load_result, "missing_keys", None):
            print(f"Warning: PyTorch reported missing keys after partial load: {load_result.missing_keys}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"Warning: PyTorch reported unexpected keys after partial load: {load_result.unexpected_keys}")

    return {
        "matched_keys": list(matched_state.keys()),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatched_keys": shape_mismatched_keys,
    }

def _parse_bool_from_train_log(log_path, key):
    if not os.path.isfile(log_path):
        return None
    pattern = re.compile(rf"{re.escape(key)}=(true|false)", re.IGNORECASE)
    value = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                value = match.group(1).lower() == "true"
    return value

def load_run_metadata(log_dir, filename=None):
    preferred_path = get_run_metadata_path(log_dir, filename=filename)
    metadata_candidates = []
    if os.path.isfile(preferred_path):
        metadata_candidates.append(preferred_path)

    glob_pattern = os.path.join(log_dir, "run_metadata*.json")
    extra_candidates = [
        path for path in glob.glob(glob_pattern)
        if os.path.isfile(path) and path not in metadata_candidates
    ]
    extra_candidates.sort(key=os.path.getmtime, reverse=True)
    metadata_candidates.extend(extra_candidates)

    if metadata_candidates:
        metadata_path = metadata_candidates[0]
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source"] = metadata_path
        return data

    train_log_path = os.path.join(log_dir, "train.log")
    checkpoint_features = {}
    observe_gait_commands = _parse_bool_from_train_log(train_log_path, "OBSERVE_GAIT_COMMANDS")
    mixed_height_reference = _parse_bool_from_train_log(train_log_path, "MIXED_HEIGHT_REFERENCE")
    omnidirectional_pos_y = _parse_bool_from_train_log(train_log_path, "OMNIDIRECTIONAL_POS_Y")
    if observe_gait_commands is not None:
        checkpoint_features["observe_gait_commands"] = observe_gait_commands
    if mixed_height_reference is not None:
        checkpoint_features["mixed_height_reference"] = mixed_height_reference
    if omnidirectional_pos_y is not None:
        checkpoint_features["omnidirectional_pos_y"] = omnidirectional_pos_y
    if checkpoint_features:
        return {
            "checkpoint_features": checkpoint_features,
            "_source": train_log_path,
        }
    return None

def apply_checkpoint_features_from_run(args, log_dir, verbose=True):
    metadata = load_run_metadata(log_dir)
    if metadata is None:
        if verbose:
            print(f"No run metadata found under: {log_dir}")
        return args, None

    checkpoint_features = metadata.get("checkpoint_features", {})
    if "observe_gait_commands" in checkpoint_features:
        args.observe_gait_commands = bool(checkpoint_features["observe_gait_commands"])
    if "mixed_height_reference" in checkpoint_features:
        args.mixed_height_reference = bool(checkpoint_features["mixed_height_reference"])
    if "omnidirectional_pos_y" in checkpoint_features:
        args.omnidirectional_pos_y = bool(checkpoint_features["omnidirectional_pos_y"])

    if verbose:
        print(
            "Loaded checkpoint features from {}: observe_gait_commands={}, mixed_height_reference={}, omnidirectional_pos_y={}".format(
                metadata.get("_source", "<unknown>"),
                getattr(args, "observe_gait_commands", False),
                getattr(args, "mixed_height_reference", False),
                getattr(args, "omnidirectional_pos_y", False),
            )
        )
    return args, metadata

def set_seed(seed, verbose=True):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    if verbose:
        print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            if getattr(args, "rank", 0) == 0:
                print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 6 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.stop_update_goal is not None:
            env_cfg.env.stop_update_goal = args.stop_update_goal
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols
        if args.observe_gait_commands:
            env_cfg.env.observe_gait_commands = True
        if args.teleop_mode:
            env_cfg.env.teleop_mode = True
        if args.teleop_input_regularization:
            env_cfg.env.teleop_input_regularization = True
        if args.action_delay_mode is not None:
            env_cfg.env.action_delay_mode = args.action_delay_mode
        if args.record_video:
            env_cfg.env.record_video = args.record_video
        if args.stand_by:
            env_cfg.env.stand_by = args.stand_by
        if args.vel_obs:
            env_cfg.env.observe_velocities = args.vel_obs
        env_cfg.env.pitch_control = args.pitch_control
        if args.mixed_height_reference:
            env_cfg.goal_ee.sphere_center.mixed_height_reference = True
        if args.trunk_follow_ratio is not None:
            env_cfg.goal_ee.sphere_center.trunk_follow_ratio = args.trunk_follow_ratio
        if args.omnidirectional_pos_y:
            env_cfg.goal_ee.ranges.omnidirectional_pos_y = True
        lin_vel_x_min_schedule = _parse_schedule_arg(args.lin_vel_x_min_schedule)
        if lin_vel_x_min_schedule is not None:
            env_cfg.commands.lin_vel_x_min_schedule = lin_vel_x_min_schedule
        lin_vel_x_max_schedule = _parse_schedule_arg(args.lin_vel_x_max_schedule)
        if lin_vel_x_max_schedule is not None:
            env_cfg.commands.lin_vel_x_max_schedule = lin_vel_x_max_schedule
        ang_vel_yaw_schedule = _parse_schedule_arg(args.ang_vel_yaw_schedule)
        if ang_vel_yaw_schedule is not None:
            env_cfg.commands.ang_vel_yaw_schedule = ang_vel_yaw_schedule
        tracking_lin_vel_max_schedule = _parse_schedule_arg(args.tracking_lin_vel_max_schedule)
        if tracking_lin_vel_max_schedule is not None:
            env_cfg.rewards.tracking_lin_vel_max_schedule = tracking_lin_vel_max_schedule
        tracking_ang_vel_schedule = _parse_schedule_arg(args.tracking_ang_vel_schedule)
        if tracking_ang_vel_schedule is not None:
            env_cfg.rewards.tracking_ang_vel_schedule = tracking_ang_vel_schedule
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        mixing_schedule = _parse_schedule_arg(args.mixing_schedule)
        if mixing_schedule is not None:
            cfg_train.algorithm.mixing_schedule = mixing_schedule
        priv_reg_coef_schedule = _parse_schedule_arg(args.priv_reg_coef_schedule)
        if priv_reg_coef_schedule is not None:
            cfg_train.algorithm.priv_reg_coef_schedule = priv_reg_coef_schedule
        else:
            priv_reg_coef_schedule_alias = _parse_schedule_arg(args.priv_reg_coef_schedual)
            if priv_reg_coef_schedule_alias is not None:
                cfg_train.algorithm.priv_reg_coef_schedule = priv_reg_coef_schedule_alias

    return env_cfg, cfg_train

def get_args(test=False):
    filtered_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--local_rank", "--local-rank"):
            skip_next = True
            continue
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            continue
        filtered_argv.append(arg)
    sys.argv = filtered_argv

    custom_parameters = [
        {"name": "--task", "type": str, "default": "widowGo1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "required": False,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "default": "", "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,"default": "-1",  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--stop_update_goal", "action": "store_true", "help": "stop when update a new ee goal"},
        {"name": "--observe_gait_commands", "action": "store_true", "help": "if observe gait commands, ref to <walk these ways>"},
        
        {"name": "--exptid", "type": str,  "required": True if not test else False,  "help": "Experiment ID"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str,  "default": "b1z1-low", "help": "run folder name."},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--load_exptid", "type": str, "help": "Checkpoint source exptid for load mode. Loads weights only and starts training from iteration 0."},
        {"name": "--train_mode", "type": str, "default": "fresh", "help": "Training mode: fresh, resume, or load."},

        {"name": "--no-headless", "action": "store_true", "help": "Enable viewer rendering"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--stochastic", "action": "store_true", "default": False, "help": "Use stochastic actions to play"},
        {"name": "--use_jit", "action": "store_true", "default": False,  "help": "Use jit to play"},
        {"name": "--record_video", "action": "store_true", "default": False,  "help": "Record video to play"},
        {"name": "--teleop_mode", "action": "store_true", "default": False,  "help": "Enable keyboard teleoperation mode"},
        {"name": "--teleop_input_regularization", "action": "store_true", "default": False, "help": "Preprocess teleop raw commands and arm targets before feeding the policy/control stack"},
        {"name": "--action_delay_mode", "type": str, "choices": ["auto", "undelayed", "delayed"], "help": "Action delay mode for play/teleop: auto keeps the training switch, undelayed always uses the latest action, delayed always uses a one-step delayed action."},
        {"name": "--stand_by", "action": "store_true", "default": False,  "help": "Stand by to play"},
        {"name": "--flat_terrain", "action": "store_true", "default": False,  "help": "Flat the terrain"},
        {"name": "--pitch_control", "action": "store_true", "default": False,  "help": "Control Pitch"},
        {"name": "--vel_obs", "action": "store_true", "default": False,  "help": "Control Pitch"},
        {"name": "--mixed_height_reference", "action": "store_true", "default": False, "help": "Train both z-invariant and trunk-height-following goal modes"},
        {"name": "--trunk_follow_ratio", "type": float, "help": "Fraction of trunk-height-following goal episodes when mixed_height_reference is enabled"},
        {"name": "--omnidirectional_pos_y", "action": "store_true", "default": False, "help": "Sample end-effector goal yaw omnidirectionally, using pos_y as a relative-yaw window"},
        {"name": "--lin_vel_x_min_schedule", "type": str, "help": "Curriculum for lin_vel_x command minimum as comma-separated values: start,end or start,end,start_iter,end_iter."},
        {"name": "--lin_vel_x_max_schedule", "type": str, "help": "Curriculum for lin_vel_x command maximum as comma-separated values: start,end or start,end,start_iter,end_iter."},
        {"name": "--ang_vel_yaw_schedule", "type": str, "help": "Curriculum for |ang_vel_yaw| command range as comma-separated values: start,end or start,end,start_iter,end_iter."},
        {"name": "--tracking_lin_vel_max_schedule", "type": str, "help": "Curriculum for rewards.scales.tracking_lin_vel_max as comma-separated values: start,end or start,end,start_iter,end_iter."},
        {"name": "--tracking_ang_vel_schedule", "type": str, "help": "Curriculum for rewards.scales.tracking_ang_vel as comma-separated values: start,end or start,end,start_iter,end_iter."},
        {"name": "--mixing_schedule", "type": str, "help": "Value mixing schedule as comma-separated values: target,start_iter,end_iter or start,end,start_iter,end_iter."},
        {"name": "--priv_reg_coef_schedule", "type": str, "help": "Privileged-reference regularization schedule as comma-separated values: start,end,start_iter,end_iter."},
        {"name": "--priv_reg_coef_schedual", "type": str, "help": "Deprecated alias for --priv_reg_coef_schedule."},
        
        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.test = test
    args.headless = not getattr(args, "no_headless", False)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    

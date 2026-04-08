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
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerHRL

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params, get_run_log_dir, load_matching_checkpoint_weights
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .logger import log_files

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        self.task_paths = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO, path: str):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
        self.task_paths[name] = path
    
    def get_task_class(self, name: str) -> VecEnv:
        self.curr_task_path = self.task_paths[name]
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            print(f'task_classes are:{self.task_classes}')
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        world_size = getattr(args, "world_size", 1)
        if world_size > 1:
            if env_cfg.env.num_envs < world_size:
                raise ValueError(f"num_envs ({env_cfg.env.num_envs}) must be >= world_size ({world_size})")
            if env_cfg.env.num_envs % world_size != 0:
                raise ValueError(f"num_envs ({env_cfg.env.num_envs}) must be divisible by world_size ({world_size}) for distributed training")
            env_cfg.env.num_envs //= world_size
        distributed_rank = getattr(args, "rank", 0)
        env_cfg.seed += distributed_rank
        set_seed(env_cfg.seed, verbose=getattr(args, "rank", 0) == 0)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", **kwargs) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        checkpoint = args.checkpoint
        if checkpoint == -1:
            checkpoint = train_cfg.runner.checkpoint
        
        log_dir = log_root
        
        train_cfg_dict = class_to_dict(train_cfg)
        if env.name == "b1z1_pick":
            runner = OnPolicyRunnerHRL(env,
                                       train_cfg_dict,
                                       log_dir,
                                       device=args.rl_device)
        else:
            runner = OnPolicyRunner(env, 
                                    train_cfg_dict, 
                                    log_dir, 
                                    device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        load_exptid = getattr(args, "load_exptid", None)
        load_only = bool(load_exptid) and not resume
        resume_path = None
        if args.resumeid:
            log_root = get_run_log_dir(args.proj_name, args.resumeid)
            resume = True
        if resume:
            # load previously trained model
            if getattr(args, "rank", 0) == 0:
                print(log_root)
            resume_path = get_load_path(log_root, checkpoint=checkpoint)
            if getattr(args, "rank", 0) == 0:
                print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            if checkpoint == -1:
                checkpoint = int(resume_path.split("_")[-1].split(".")[0])
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)
        elif load_only:
            load_root = get_run_log_dir(args.proj_name, load_exptid)
            load_path = get_load_path(load_root, checkpoint=checkpoint)
            if getattr(args, "rank", 0) == 0:
                print(f"Loading model weights from: {load_path}")
            load_matching_checkpoint_weights(
                runner.alg.actor_critic,
                load_path,
                device=runner.device,
                verbose=getattr(args, "rank", 0) == 0,
            )
            runner.set_it(0)
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)
            if checkpoint == -1:
                checkpoint = int(load_path.split("_")[-1].split(".")[0])
            resume_path = load_path

        if "return_log_dir" in kwargs:
            return runner, train_cfg, checkpoint, os.path.dirname(resume_path) if resume_path else log_dir
        else:    
            return runner, train_cfg, checkpoint

# make global task registry
task_registry = TaskRegistry()

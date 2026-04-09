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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_schedule_value

import wandb
from torchinfo import summary

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main_process = self.rank == 0
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.cfg.env.num_proprio,
                                                        self.env.cfg.env.num_proprio,
                                                        self.env.num_actions,
                                                        **self.policy_cfg, 
                                                        num_priv=env.cfg.env.num_priv,
                                                        num_hist=env.cfg.env.history_len, 
                                                        num_prop=env.cfg.env.num_proprio,
                                                        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.train_log_every = max(1, int(self.cfg.get("train_log_every", 1)))
        if self.is_main_process:
            summary(self.alg.actor_critic)

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
        self.alg.setup_distributed()

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        self.curriculum_state = {}

        _, _ = self.env.reset()

        self.alg.set_arm_default_coeffs(self.env.p_gains[12:], self.env.d_gains[12:], self.env.default_dof_pos[-7:-2])

    def _apply_env_curricula(self, iteration, total_iterations):
        self.curriculum_state = {}

        lin_vel_x_min = resolve_schedule_value(self.env.cfg.commands.lin_vel_x_min_schedule, iteration, default_end_iter=total_iterations)
        lin_vel_x_max = resolve_schedule_value(self.env.cfg.commands.lin_vel_x_max_schedule, iteration, default_end_iter=total_iterations)
        self.env.command_ranges["lin_vel_x"] = [lin_vel_x_min, lin_vel_x_max]
        self.curriculum_state["Loss/lin_vel_x_command_min"] = lin_vel_x_min
        self.curriculum_state["Loss/lin_vel_x_command_max"] = lin_vel_x_max

        ang_vel_yaw_max = resolve_schedule_value(self.env.cfg.commands.ang_vel_yaw_schedule, iteration, default_end_iter=total_iterations)
        ang_vel_yaw_min = -ang_vel_yaw_max
        self.env.command_ranges["ang_vel_yaw"] = [ang_vel_yaw_min, ang_vel_yaw_max]
        self.curriculum_state["Loss/ang_vel_yaw_command_min"] = ang_vel_yaw_min
        self.curriculum_state["Loss/ang_vel_yaw_command_max"] = ang_vel_yaw_max

        tracking_lin_vel_max_schedule = getattr(self.env.cfg.rewards, "tracking_lin_vel_max_schedule", None)
        if tracking_lin_vel_max_schedule is not None:
            tracking_lin_vel_max_scale = resolve_schedule_value(tracking_lin_vel_max_schedule, iteration, default_end_iter=total_iterations)
            self.env.reward_scales["tracking_lin_vel_max"] = tracking_lin_vel_max_scale
        self.curriculum_state["Loss/tracking_lin_vel_max_scale"] = float(self.env.reward_scales["tracking_lin_vel_max"])

        tracking_ang_vel_schedule = getattr(self.env.cfg.rewards, "tracking_ang_vel_schedule", None)
        if tracking_ang_vel_schedule is not None:
            tracking_ang_vel_scale = resolve_schedule_value(tracking_ang_vel_schedule, iteration, default_end_iter=total_iterations)
            self.env.reward_scales["tracking_ang_vel"] = tracking_ang_vel_scale
        self.curriculum_state["Loss/tracking_ang_vel_scale"] = float(self.env.reward_scales["tracking_ang_vel"])
    
    def set_it(self, it):
        self.current_learning_iteration = it
        if hasattr(self.alg, "counter"):
            self.alg.counter = it
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init metrics
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_arm_torques_loss = 0.
        value_mixing_ratio = 0.
        torque_supervision_weight = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.

        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        armrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self._apply_env_curricula(it, tot_iter)
            # self.env.update_command_curriculum()

            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, hist_encoding)
                    obs, privileged_obs, rewards, arm_rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, arm_rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), arm_rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, arm_rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_arm_reward_sum += arm_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        armrewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # self.alg.storage.clear()
            
            # mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                mean_hist_latent_loss = self.alg.update_dagger()
            else:
                mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.is_main_process:
                should_print = ((it - self.current_learning_iteration) % self.train_log_every == 0) or (it == tot_iter - 1)
                self.log(locals(), print_to_stdout=should_print)
            if self.is_main_process and it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, 'model_{}.pt'.format(it)),
                    it,
                    next_learning_iteration=it + 1,
                )
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.is_main_process:
            self.save(
                os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
                self.current_learning_iteration,
                next_learning_iteration=self.current_learning_iteration,
            )

    def log(self, locs, width=80, pad=35, print_to_stdout=True):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb.log({'Episode/' + key: value}, step=locs['it'])
                if "rew" in key:
                    wandb_dict['Episode_rew/' + key] = value
                elif "metric" in key:
                    wandb_dict['Episode_metric/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        leg_mean_std = self.alg.actor_critic.std[:, :12].mean()
        arm_mean_std = self.alg.actor_critic.std[:, 12:].mean()
        std_numpy = self.alg.actor_critic.std.cpu().detach().numpy()
        fps = int(self.num_steps_per_env * self.env.num_envs * self.world_size / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/arm_torques_loss'] = locs['mean_arm_torques_loss']
        wandb_dict['Loss/value_mixing_ratio'] = locs['value_mixing_ratio']
        wandb_dict['Loss/torque_supervision_weight'] = locs['torque_supervision_weight']
        wandb_dict.update(self.curriculum_state)
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/leg_mean_noise_std'] = leg_mean_std.item()
        wandb_dict['Policy/arm_mean_noise_std'] = arm_mean_std.item()
        wandb_dict['Policy/noise_std_dist'] = wandb.Histogram(std_numpy)
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_arm_reward'] = statistics.mean(locs['armrewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/dones'] = statistics.mean(locs['donebuffer'])
            # wandb.log({'Train/mean_reward/time': statistics.mean(locs['rewbuffer'])}, step=self.tot_time)
            # wandb.log({'Train/mean_episode_length/time': statistics.mean(locs['lenbuffer'])}, step=self.tot_time)
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                          f"""{'Privileged info regularizer lambda:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Dones:':>{pad}} {statistics.mean(locs['donebuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                          f"""{'Leg mean action noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                          f"""{'Arm mean action noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if print_to_stdout:
            print(log_string)

    def save(self, path, it, infos=None, next_learning_iteration=None):
        if next_learning_iteration is None:
            next_learning_iteration = it
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'hist_encoder_optimizer_state_dict': self.alg.hist_encoder_optimizer.state_dict(),
            'iter': it,
            'next_learning_iteration': next_learning_iteration,
            'alg_counter': getattr(self.alg, 'counter', next_learning_iteration),
            'learning_rate': getattr(self.alg, 'learning_rate', None),
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if 'hist_encoder_optimizer_state_dict' in loaded_dict:
                self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict['hist_encoder_optimizer_state_dict'])
            if 'learning_rate' in loaded_dict and loaded_dict['learning_rate'] is not None:
                self.alg.learning_rate = loaded_dict['learning_rate']
                for param_group in self.alg.optimizer.param_groups:
                    param_group['lr'] = self.alg.learning_rate
                for param_group in self.alg.hist_encoder_optimizer.param_groups:
                    param_group['lr'] = self.alg.learning_rate
        next_learning_iteration = loaded_dict.get('next_learning_iteration', loaded_dict['iter'])
        self.current_learning_iteration = next_learning_iteration
        if hasattr(self.alg, "counter"):
            self.alg.counter = loaded_dict.get('alg_counter', next_learning_iteration)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)

        if not stochastic:
            return self.alg.actor_critic.act_inference
        else:
            return self.alg.actor_critic.act

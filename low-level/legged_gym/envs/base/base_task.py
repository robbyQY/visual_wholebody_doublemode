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

import inspect
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
import time
import torch.distributed as dist

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.is_main_process = self.rank == 0

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.arm_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.manual_reset_requested = False

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            self._create_viewer()

        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec_local = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        self.lookat_follow_yaw = 0.0
        self.camera_follow_yaw_update_threshold = np.deg2rad(20.0)
        self.camera_orbit_radius_min = 0.5
        self.camera_orbit_pitch_limit = np.deg2rad(85.0)
        self.camera_orbit_yaw_step = np.deg2rad(5.0)
        self.camera_orbit_pitch_step = np.deg2rad(5.0)
        self.camera_orbit_radius_step = 0.2
        action_repeat_delay_s = getattr(
            cfg.env,
            "action_repeat_delay_s",
            getattr(cfg.env, "teleop_key_repeat_delay_s", 0.35),
        )
        action_repeat_rate_hz = getattr(
            cfg.env,
            "action_repeat_rate_hz",
            getattr(cfg.env, "teleop_key_repeat_rate_hz", 6.0),
        )
        self.action_repeat_delay_s = max(0.0, float(action_repeat_delay_s))
        self.action_repeat_rate_hz = max(0.0, float(action_repeat_rate_hz))
        self.action_repeat_period_s = (
            1.0 / self.action_repeat_rate_hz if self.action_repeat_rate_hz > 0.0 else float("inf")
        )
        self.repeatable_actions = {
            "prev_id",
            "next_id",
            "camera_orbit_left",
            "camera_orbit_right",
            "camera_orbit_up",
            "camera_orbit_down",
            "camera_orbit_zoom_in",
            "camera_orbit_zoom_out",
        }
        self.held_actions = {}

    def _create_viewer(self):
        camera_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        self.subscribe_viewer_keyboard_events()

    def subscribe_viewer_keyboard_events(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_F, "free_cam")
        for i in range(min(9, self.num_envs)):
            self.gym.subscribe_viewer_keyboard_event(
            self.viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_9, "reset_all")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SPACE, "pause")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_LEFT, "camera_orbit_left")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_RIGHT, "camera_orbit_right")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_UP, "camera_orbit_up")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "camera_orbit_down")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_PAGE_UP, "camera_orbit_zoom_in")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_PAGE_DOWN, "camera_orbit_zoom_out")


    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_all_envs(start=True)
        obs, privileged_obs, *_ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def request_manual_reset(self):
        self.manual_reset_requested = True

    def consume_manual_reset_request(self):
        requested = self.manual_reset_requested
        self.manual_reset_requested = False
        return requested

    def reset_all_envs(self, start=True):
        env_ids = torch.arange(self.num_envs, device=self.device)
        reset_signature = inspect.signature(self.reset_idx)
        if "start" in reset_signature.parameters:
            self.reset_idx(env_ids, start=start)
        else:
            self.reset_idx(env_ids)

    def step(self, actions):
        raise NotImplementedError

    def lookat(self, i):
        if i < 0 or i >= self.num_envs:
            return
        self._maybe_update_follow_yaw_anchor(i)
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self._follow_vector_local_to_world(
            self.lookat_vec_local,
            i,
            yaw=self.lookat_follow_yaw,
        )
        self.set_camera(cam_pos, look_at_pos)

    def _get_follow_yaw(self, i):
        base_quat = self.root_states[i, 3:7]
        x, y, z, w = [float(v) for v in base_quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _wrap_to_pi(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _reset_follow_yaw_anchor(self, i):
        self.lookat_follow_yaw = self._get_follow_yaw(i)

    def _maybe_update_follow_yaw_anchor(self, i):
        current_yaw = self._get_follow_yaw(i)
        yaw_error = self._wrap_to_pi(current_yaw - self.lookat_follow_yaw)
        if abs(yaw_error) >= self.camera_follow_yaw_update_threshold:
            self.lookat_follow_yaw = current_yaw

    def _follow_vector_local_to_world(self, vec_local, i, yaw=None):
        yaw = self._get_follow_yaw(i) if yaw is None else yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return torch.tensor(
            [
                cos_yaw * vec_local[0].item() - sin_yaw * vec_local[1].item(),
                sin_yaw * vec_local[0].item() + cos_yaw * vec_local[1].item(),
                vec_local[2].item(),
            ],
            requires_grad=False,
            device=self.device,
            dtype=torch.float,
        )

    def _follow_vector_world_to_local(self, vec_world, i, yaw=None):
        yaw = self._get_follow_yaw(i) if yaw is None else yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return torch.tensor(
            [
                cos_yaw * vec_world[0].item() + sin_yaw * vec_world[1].item(),
                -sin_yaw * vec_world[0].item() + cos_yaw * vec_world[1].item(),
                vec_world[2].item(),
            ],
            requires_grad=False,
            device=self.device,
            dtype=torch.float,
        )

    def _get_viewer_camera_position(self):
        cam_transform = self.gym.get_viewer_camera_transform(self.viewer, None).p
        return torch.tensor(
            [cam_transform.x, cam_transform.y, cam_transform.z],
            requires_grad=False,
            device=self.device,
            dtype=torch.float,
        )

    def _orbit_camera(self, delta_yaw=0.0, delta_pitch=0.0, delta_radius=0.0):
        radius = torch.norm(self.lookat_vec_local).item()
        if radius < 1e-6:
            radius = 1.0

        yaw = np.arctan2(self.lookat_vec_local[1].item(), self.lookat_vec_local[0].item())
        pitch = np.arcsin(np.clip(self.lookat_vec_local[2].item() / radius, -1.0, 1.0))

        radius = max(self.camera_orbit_radius_min, radius + delta_radius)
        pitch = np.clip(pitch + delta_pitch, -self.camera_orbit_pitch_limit, self.camera_orbit_pitch_limit)
        yaw = yaw + delta_yaw

        cos_pitch = np.cos(pitch)
        self.lookat_vec_local = torch.tensor(
            [
                radius * cos_pitch * np.cos(yaw),
                radius * cos_pitch * np.sin(yaw),
                radius * np.sin(pitch),
            ],
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.lookat(self.lookat_id)

    def _register_repeatable_actions(self, *actions):
        self.repeatable_actions.update(actions)

    def _update_held_action_state(self, action, value):
        if action not in self.repeatable_actions:
            return

        if value <= 0:
            self.held_actions.pop(action, None)
            return

        if action not in self.held_actions:
            now = time.monotonic()
            self.held_actions[action] = {
                "next_repeat_time": now + self.action_repeat_delay_s,
            }

    def _apply_held_actions(self):
        if not self.held_actions or self.action_repeat_rate_hz <= 0.0:
            return

        now = time.monotonic()
        for action, state in list(self.held_actions.items()):
            if now < state["next_repeat_time"]:
                continue
            self.handle_repeated_action(action)
            state["next_repeat_time"] = now + self.action_repeat_period_s

    def _apply_base_viewer_action(self, action):
        if action == "QUIT":
            sys.exit()
        elif action == "toggle_viewer_sync":
            self.enable_viewer_sync = not self.enable_viewer_sync
            return True
        elif action == "reset_all":
            self.request_manual_reset()
            print("[viewer] manual reset requested for all environments")
            return True

        if not self.free_cam:
            for i in range(min(9, self.num_envs)):
                if action == "lookat" + str(i):
                    self._reset_follow_yaw_anchor(i)
                    self.lookat(i)
                    self.lookat_id = i
                    return True
            if action == "prev_id":
                self.lookat_id  = (self.lookat_id-1) % self.num_envs
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat(self.lookat_id)
                return True
            if action == "next_id":
                self.lookat_id  = (self.lookat_id+1) % self.num_envs
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat(self.lookat_id)
                return True
            if action == "camera_orbit_left":
                self._orbit_camera(delta_yaw=-self.camera_orbit_yaw_step)
                return True
            if action == "camera_orbit_right":
                self._orbit_camera(delta_yaw=self.camera_orbit_yaw_step)
                return True
            if action == "camera_orbit_up":
                self._orbit_camera(delta_pitch=self.camera_orbit_pitch_step)
                return True
            if action == "camera_orbit_down":
                self._orbit_camera(delta_pitch=-self.camera_orbit_pitch_step)
                return True
            if action == "camera_orbit_zoom_in":
                self._orbit_camera(delta_radius=-self.camera_orbit_radius_step)
                return True
            if action == "camera_orbit_zoom_out":
                self._orbit_camera(delta_radius=self.camera_orbit_radius_step)
                return True

        if action == "free_cam":
            self.free_cam = not self.free_cam
            if not self.free_cam:
                # Re-enter follow mode from the current free-camera position so the
                # transition stays continuous instead of snapping to a preset view.
                cam_trans = self._get_viewer_camera_position()
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self._reset_follow_yaw_anchor(self.lookat_id)
                self.lookat_vec_local = self._follow_vector_world_to_local(
                    cam_trans - look_at_pos,
                    self.lookat_id,
                    yaw=self.lookat_follow_yaw,
                )
            return True
        
        if action == "pause":
            self.pause = True
            while self.pause:
                time.sleep(0.1)
                self._draw_viewer()
                for evt in self.gym.query_viewer_action_events(self.viewer):
                    if evt.action == "pause" and evt.value > 0:
                        self.pause = False
            return True

        return False

    def handle_repeated_action(self, action):
        return self._apply_base_viewer_action(action)

    def handle_viewer_action_event(self, evt):
        self._update_held_action_state(evt.action, evt.value)
        if evt.value <= 0:
            return
        self._apply_base_viewer_action(evt.action)

    def on_viewer_events_processed(self):
        self._apply_held_actions()

    def _draw_viewer(self):
        self.gym.draw_viewer(
            self.viewer,
            self.sim,
            False,
        )

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if not self.free_cam:
                self.lookat(self.lookat_id)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                self.handle_viewer_action_event(evt)
            self.on_viewer_events_processed()

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self._draw_viewer()
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

            if not self.free_cam:
                cam_trans = self._get_viewer_camera_position()
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self.lookat_vec_local = self._follow_vector_world_to_local(
                    cam_trans - look_at_pos,
                    self.lookat_id,
                    yaw=self.lookat_follow_yaw,
                )

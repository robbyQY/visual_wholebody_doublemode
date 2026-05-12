# IsaacGym -> IsaacLab migration inventory

下表列出当前 `legged_gym` 中直接依赖 IsaacGym API 的代码行，供迁移到 IsaacLab/Isaac Sim 使用。

| File | Line | IsaacGym usage | IsaacLab migration note |
|---|---:|---|---|
| `low-level/legged_gym/utils/terrain.py` | 36 | `from isaacgym import terrain_utils` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 57 | `self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples,` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 139 | `self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 179 | `terrain = terrain_utils.SubTerrain("terrain",` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 191 | `terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 194 | `terrain = terrain_utils.SubTerrain(   "terrain",` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 209 | `terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 214 | `terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 219 | `terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 225 | `terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 229 | `terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 230 | `# terrain_utils.stepping_stones_terrain(terrain, stone_size=0.2-0.06*difficulty, stone_distance=0.06+0.06*difficulty, stone_distance_rand=0.06*difficulty, max_height=0.04*difficulty, platform_size=2.)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 256 | `# terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/terrain.py` | 257 | `# terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)` | Use IsaacLab terrain importer/generator utils. |
| `low-level/legged_gym/utils/math.py` | 34 | `from isaacgym.torch_utils import quat_apply, normalize` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/helpers.py` | 40 | `from isaacgym import gymapi` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/helpers.py` | 41 | `from isaacgym import gymutil` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/helpers.py` | 528 | `sim_params = gymapi.SimParams()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/utils/helpers.py` | 531 | `if args.physics_engine == gymapi.SIM_FLEX:` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/utils/helpers.py` | 535 | `elif args.physics_engine == gymapi.SIM_PHYSX:` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/utils/helpers.py` | 542 | `gymutil.parse_sim_config(cfg["sim"], sim_params)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/helpers.py` | 545 | `if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/utils/helpers.py` | 850 | `args = gymutil.parse_arguments(` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/task_registry.py` | 81 | `isaacgym.VecTaskPython: The created environment` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/utils/task_registry.py` | 122 | `env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/train.py` | 37 | `import isaacgym` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/manip_loco_interface.py` | 4 | `import isaacgym` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/manip_loco_interface.py` | 5 | `from isaacgym.torch_utils import euler_from_quat` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/play.py` | 34 | `import isaacgym` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/play.py` | 35 | `from isaacgym import gymtorch` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/play.py` | 36 | `from isaacgym.torch_utils import euler_from_quat, quat_apply` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/scripts/play.py` | 254 | `env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/scripts/play.py` | 258 | `env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/scripts/play.py` | 286 | `env.gym.refresh_actor_root_state_tensor(env.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/scripts/play.py` | 287 | `env.gym.refresh_dof_state_tensor(env.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/scripts/play.py` | 288 | `env.gym.refresh_rigid_body_state_tensor(env.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/scripts/play.py` | 289 | `env.gym.refresh_jacobian_tensors(env.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/tests/test_env.py` | 35 | `import isaacgym` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/rewards/maniploco_rewards.py` | 2 | `from isaacgym.torch_utils import *` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 37 | `from isaacgym.torch_utils import *` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 38 | `from isaacgym import gymtorch, gymapi, gymutil` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 54 | `calls create_sim() (which creates, simulation, terrain and environments),` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 59 | `sim_params (gymapi.SimParams): simulation parameters` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 60 | `physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 96 | `self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 97 | `self.gym.simulate(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 99 | `self.gym.fetch_results(self.sim, True)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 100 | `self.gym.refresh_dof_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 115 | `self.gym.refresh_actor_root_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 116 | `self.gym.refresh_net_contact_force_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 233 | `def create_sim(self):` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 237 | `self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 254 | `cam_pos = gymapi.Vec3(position[0], position[1], position[2])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 255 | `cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 265 | `props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 269 | `[List[gymapi.RigidShapeProperties]]: Modified rigid shape properties` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 395 | `gymtorch.unwrap_tensor(self.dof_state),` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 396 | `gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 416 | `gymtorch.unwrap_tensor(self.root_states),` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 417 | `gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 424 | `self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 537 | `self.gym.refresh_dof_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 538 | `self.gym.refresh_actor_root_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 539 | `self.gym.refresh_net_contact_force_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 542 | `self.root_states = gymtorch.wrap_tensor(actor_root_state)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 543 | `self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 548 | `self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 621 | `plane_params = gymapi.PlaneParams()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 622 | `plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 632 | `hf_params = gymapi.HeightFieldProperties()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 651 | `tm_params = gymapi.TriangleMeshParams()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 677 | `asset_options = gymapi.AssetOptions()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 724 | `start_pose = gymapi.Transform()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 725 | `start_pose.p = gymapi.Vec3(*self.base_init_state[:3])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 728 | `env_lower = gymapi.Vec3(0., 0., 0.)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 729 | `env_upper = gymapi.Vec3(0., 0., 0.)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 737 | `start_pose.p = gymapi.Vec3(*pos)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 809 | `self.gym.refresh_rigid_body_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 810 | `sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/legged_robot.py` | 819 | `sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/legged_robot.py` | 820 | `gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 33 | `from isaacgym import gymapi` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/base_task.py` | 34 | `from isaacgym import gymutil` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/base_task.py` | 44 | `self.gym = gymapi.acquire_gym()` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/base/base_task.py` | 52 | `sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/base_task.py` | 91 | `self.create_sim()` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/base/base_task.py` | 141 | `camera_props = gymapi.CameraProperties()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/base/base_task.py` | 147 | `self.viewer, gymapi.KEY_ESCAPE, "QUIT")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 149 | `self.viewer, gymapi.KEY_V, "toggle_viewer_sync")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 151 | `self.viewer, gymapi.KEY_F, "free_cam")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 154 | `self.viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 156 | `self.viewer, gymapi.KEY_9, "reset_all")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 158 | `self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 160 | `self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 162 | `self.viewer, gymapi.KEY_SPACE, "pause")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 164 | `self.viewer, gymapi.KEY_LEFT, "camera_orbit_left")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 166 | `self.viewer, gymapi.KEY_RIGHT, "camera_orbit_right")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 168 | `self.viewer, gymapi.KEY_UP, "camera_orbit_up")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 170 | `self.viewer, gymapi.KEY_DOWN, "camera_orbit_down")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 172 | `self.viewer, gymapi.KEY_PAGE_UP, "camera_orbit_zoom_in")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 174 | `self.viewer, gymapi.KEY_PAGE_DOWN, "camera_orbit_zoom_out")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 401 | `self._draw_viewer()` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 421 | `def _draw_viewer(self):` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 422 | `self.gym.draw_viewer(` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/base/base_task.py` | 442 | `self.gym.fetch_results(self.sim, True)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/base/base_task.py` | 447 | `self.gym.step_graphics(self.sim)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/base/base_task.py` | 448 | `self._draw_viewer()` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 35 | `from isaacgym.torch_utils import *` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 36 | `from isaacgym import gymtorch, gymapi, gymutil` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 107 | `self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 108 | `self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 109 | `self.gym.simulate(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 111 | `self.gym.fetch_results(self.sim, True)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 112 | `self.gym.refresh_dof_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 113 | `self.gym.refresh_jacobian_tensors(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 114 | `self.gym.refresh_rigid_body_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 139 | `self.gym.refresh_dof_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 140 | `self.gym.refresh_actor_root_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 141 | `self.gym.refresh_net_contact_force_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 142 | `self.gym.refresh_force_sensor_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 143 | `self.gym.refresh_rigid_body_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 144 | `self.gym.refresh_jacobian_tensors(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 319 | `def create_sim(self):` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 323 | `self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)` | Move to SimulationCfg + InteractiveScene setup. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 488 | `plane_params = gymapi.PlaneParams()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 489 | `plane_params.normal = gymapi.Vec3(0, 0, 1)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 502 | `tm_params = gymapi.TriangleMeshParams()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 532 | `asset_options = gymapi.AssetOptions()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 553 | `dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 595 | `sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 602 | `asset_options = gymapi.AssetOptions()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 621 | `start_pose = gymapi.Transform()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 622 | `start_pose.p = gymapi.Vec3(*self.base_init_state[:3])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 623 | `box_start_pose = gymapi.Transform()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 626 | `env_lower = gymapi.Vec3(0., 0., 0.)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 627 | `env_upper = gymapi.Vec3(0., 0., 0.)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 640 | `rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.init_state.rand_yaw_range*np.random.uniform(-1, 1))` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 642 | `start_pose.p = gymapi.Vec3(*pos)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 661 | `box_start_pose.p = gymapi.Vec3(*box_pos)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 705 | `camera_props = gymapi.CameraProperties()` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 715 | `self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 737 | `props[self.base_body_idx].com += gymapi.Vec3(*rand_com)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 760 | `props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 764 | `[List[gymapi.RigidShapeProperties]]: Modified rigid shape properties` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 797 | `self.gym.refresh_dof_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 798 | `self.gym.refresh_actor_root_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 799 | `self.gym.refresh_net_contact_force_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 800 | `self.gym.refresh_rigid_body_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 801 | `self.gym.refresh_jacobian_tensors(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 802 | `self.gym.refresh_force_sensor_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 805 | `self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 806 | `self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13) # 2 actors` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 809 | `self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 836 | `self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 840 | `self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies + 1, 13)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 844 | `self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1075 | `self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1076 | `self.gym.refresh_actor_root_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1088 | `self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1116 | `self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))` | Use ArticulationData tensors and direct torch buffers in IsaacLab. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1117 | `self.gym.refresh_rigid_body_state_tensor(self.sim)` | Use env.step(actions) and scene.update(sim_dt) data flow. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1490 | `sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1495 | `bbox_geom = gymutil.WireframeBBoxGeometry(bboxes[i], None, color=(1, 0, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1497 | `r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1498 | `pose0 = gymapi.Transform(gymapi.Vec3(goal_ref_origin[i, 0], goal_ref_origin[i, 1], goal_ref_origin[i, 2]), r=r)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1499 | `gymutil.draw_lines(bbox_geom, self.gym, self.viewer, self.envs[i], pose=pose0)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1512 | `sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1513 | `sphere_geom_root = gymutil.WireframeSphereGeometry(0.06, 16, 16, None, color=(1, 1, 1))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1515 | `sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 16, 16, None, color=(0, 1, 1))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1518 | `sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1521 | `sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1522 | `sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1524 | `gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[0], sphere_pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1526 | `axes_geom = gymutil.AxesGeometry(scale=0.2)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1529 | `sphere_pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1530 | `gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1532 | `sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1533 | `gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1535 | `sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1536 | `gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1538 | `root_pose = gymapi.Transform(` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1539 | `gymapi.Vec3(self.root_states[i, 0], self.root_states[i, 1], self.root_states[i, 2]),` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1540 | `r=gymapi.Quat(self.base_quat[i, 0], self.base_quat[i, 1], self.base_quat[i, 2], self.base_quat[i, 3])` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1542 | `gymutil.draw_lines(sphere_geom_root, self.gym, self.viewer, self.envs[i], root_pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1543 | `gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], root_pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1545 | `pose = gymapi.Transform(gymapi.Vec3(self.curr_ee_goal_cart_world[i, 0], self.curr_ee_goal_cart_world[i, 1], self.curr_ee_goal_cart_world[i, 2]),` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1546 | `r=gymapi.Quat(self.ee_goal_orn_quat[i, 0], self.ee_goal_orn_quat[i, 1], self.ee_goal_orn_quat[i, 2], self.ee_goal_orn_quat[i, 3]))` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1547 | `gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1554 | `sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1555 | `sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 1, 0))` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1572 | `pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[j, i, 0], ee_target_all_cart_world[j, i, 1], ee_target_all_cart_world[j, i, 2]), r=None)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1573 | `gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1841 | `self.gym.step_graphics(self.sim)` | Replace with IsaacLab scene/sim APIs. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1847 | `self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1854 | `img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)` | Replace structures with IsaacLab cfg classes (RigidBodyPropertiesCfg, AssetBaseCfg, etc.). |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1865 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "forward")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1866 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "reverse")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1867 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "turn_left")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1868 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "turn_right")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1869 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "stop_linear")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1870 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "stop_angular")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1872 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "increase_eef_goal_l")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1873 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "decrease_eef_goal_l")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1874 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "increase_eef_goal_p")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1875 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "decrease_eef_goal_p")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1876 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "increase_eef_goal_y")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1877 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "decrease_eef_goal_y")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1879 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Z, "increase_eef_goal_dr")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1880 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "decrease_eef_goal_dr")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1881 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "increse_eef_goal_dp")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1882 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "decrease_eef_goal_dp")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1883 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "increase_eef_goal_dy")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1884 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "decrease_eef_goal_dy")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1885 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "open_gripper")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1886 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "close_gripper")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1887 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "reset_eef_goal_pose")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1888 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "toggle_arm_control_mode")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1890 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "set_height_reference_invariant")` | Use IsaacLab app loop / viewport utilities. |
| `low-level/legged_gym/envs/manip_loco/manip_loco.py` | 1891 | `self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "set_height_reference_follow")` | Use IsaacLab app loop / viewport utilities. |
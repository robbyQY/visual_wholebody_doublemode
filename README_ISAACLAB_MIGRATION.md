# legged_gym -> IsaacSim/IsaacLab migration package

This zip keeps your original directory layout but replaces the core IsaacGym simulator calls with an IsaacLab DirectRLEnv skeleton.

## What changed

- `envs/base/base_task.py`: old `gymapi.acquire_gym()` task base is removed; IsaacLab owns stepping through `DirectRLEnv`.
- `envs/base/legged_robot_isaaclab.py`: replaces `gym.create_sim`, `gym.create_env`, `gym.create_actor`, `gym.simulate`, `gymtorch.wrap_tensor`, `set_dof_actuation_force_tensor`, `set_dof_state_tensor_indexed`, and root-state writes with IsaacLab `Articulation` APIs.
- `envs/manip_loco/manip_loco_isaaclab.py`: B2/Z1-shaped subclass for initial simulator bring-up.
- `envs/manip_loco/b2z1_isaaclab_config.py`: carries your B2Z1 joint names, nominal joint angles, PD gains, foot names, and body names.
- `utils/isaaclab_math.py`: local `wxyz` quaternion math replacement for `isaacgym.torch_utils`.
- Original IsaacGym files are preserved with `_legacy_isaacgym.py` suffix.

## First run

Convert your URDF to USD first, then run:

```bash
cd <this zip root>
export PYTHONPATH=$PWD:$PYTHONPATH
./isaaclab.sh -p legged_gym/scripts/play_isaaclab.py   --robot_usd_path /abs/path/to/b2z1.usd   --num_envs 1
```

The first validation target is not policy quality. It is:

1. USD loads.
2. Printed `joint_names` match `B2Z1IsaacLabCfg.policy_joint_names`.
3. Printed `body_names` contain foot and base/gripper names.
4. Zero action does not explode.

## Known incomplete parts

The following original IsaacGym features are intentionally not fully ported in this first pass:

- rough terrain trimesh generation through `gym.add_triangle_mesh`; currently plane terrain is used first;
- Gym force sensors; use `ContactSensor` and validate foot body regex/body paths;
- Jacobian-based Z1 IK in `ManipLoco`; re-enable after gripper body name and Jacobian indexing are verified;
- box actor/object manipulation rewards;
- old `task_registry` and old RSL-RL runner entry points.

This is a simulator-interface migration package, not a guaranteed one-shot reproduction of the entire training stack.

legged_gym/
├── scripts/
│   ├── train.py                  # old IsaacGym train, 暂时不动
│   ├── play.py                   # old IsaacGym play, 暂时不动
│   └── play_isaaclab.py          # new IsaacLab play, 新增一个
│
├── envs/
│   ├── __init__.py               # 修改：IsaacLab-only 时不 import legacy isaacgym env
│   ├── base/
│   │   ├── base_config.py        # 不动
│   │   ├── base_task.py          # old IsaacGym, 不动但 IsaacLab 不用
│   │   ├── legged_robot_config.py# old IsaacGym, 不动
│   │   ├── legged_robot.py       # old IsaacGym, 不动
│   │   ├── legged_robot_isaaclab_config.py  # new
│   │   └── legged_robot_isaaclab.py         # new
│   │
│   └── manip_loco/
│       ├── b2z1_config.py        # old config，保留作 reference
│       ├── manip_loco.py         # old IsaacGym env，保留
│       ├── manip_loco_base_config.py # old config，保留
│       ├── b2z1_isaaclab_config.py   # new
│       └── manip_loco_isaaclab.py    # new
│
└── utils/
    ├── __init__.py               # 修改：不要自动 import terrain/task_registry
    ├── math.py                   # 如果仍 import isaacgym，需要换成 isaaclab/torch implementation
    └── isaaclab_math.py          # 可选，如果你不想改 math.py
"""IsaacLab migration note.

The original BaseTask was an IsaacGym wrapper around gymapi.acquire_gym(), create_sim(),
and gymtorch tensor acquisition. IsaacLab owns simulation stepping through DirectRLEnv,
so tasks should inherit from isaaclab.envs.DirectRLEnv instead of this class.

This file is kept only so old imports fail with a clear message.
"""

class BaseTask:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "BaseTask is IsaacGym-only. Use legged_gym.envs.manip_loco.manip_loco_isaaclab.ManipLocoIsaacLab "
            "or legged_gym.envs.base.legged_robot_isaaclab.LeggedRobotIsaacLab instead."
        )

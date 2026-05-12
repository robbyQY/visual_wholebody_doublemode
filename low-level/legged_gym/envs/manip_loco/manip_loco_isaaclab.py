from __future__ import annotations
import torch
from legged_gym.envs.base.legged_robot_isaaclab import LeggedRobotIsaacLab
from .b2z1_isaaclab_config import B2Z1IsaacLabCfg


class ManipLocoIsaacLab(LeggedRobotIsaacLab):
    """IsaacLab simulator-layer port for the uploaded ManipLoco task.

    This class keeps the old B2/Z1 action dimension and core simulator operations.
    The original IK/Jacobian/box-object reward stack is preserved in legacy files and
    should be migrated after the USD body names and Jacobian APIs are verified.
    """
    cfg: B2Z1IsaacLabCfg

    def _pre_physics_step(self, actions: torch.Tensor):
        # old ManipLoco zeroed arm action columns before converting policy->env.
        actions = actions.clone()
        if actions.shape[-1] > 12:
            # keep arm actions available but bounded; set to zero here to avoid unstable first bring-up.
            # Remove this line after validating arm joint order and IK/Jacobian migration.
            actions[:, 12:] = 0.0
        super()._pre_physics_step(actions)

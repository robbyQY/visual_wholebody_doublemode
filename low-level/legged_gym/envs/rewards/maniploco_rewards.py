# IsaacLab-safe subset. The full IsaacGym reward implementation is saved as maniploco_rewards_legacy.py.
import torch
from legged_gym.utils.isaaclab_math import *

class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env
    def load_env(self, env):
        self.env = env
    def _reward_tracking_lin_vel(self):
        err = torch.sum((self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]) ** 2, dim=1)
        return torch.exp(-err / 0.25), torch.sqrt(err)
    def _reward_tracking_ang_vel(self):
        err = (self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]) ** 2
        return torch.exp(-err / 0.25), torch.sqrt(err)
    def _reward_torques(self):
        val = torch.sum(self.env.torques ** 2, dim=1)
        return val, torch.sqrt(val)
    def _reward_action_rate(self):
        val = torch.sum((self.env.actions - self.env.last_actions) ** 2, dim=1)
        return val, torch.sqrt(val)
    def _reward_termination(self):
        val = self.env.reset_buf.float()
        return val, val

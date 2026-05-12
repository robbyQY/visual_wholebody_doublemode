"""Environment package initialization.

This repository originally registered IsaacGym tasks from this file.
When running the IsaacLab migration scripts, IsaacGym is not installed, so
importing the legacy task modules here would fail before the IsaacLab modules
can even be imported.
"""

from __future__ import annotations

_ISAACGYM_AVAILABLE = False
try:
    import isaacgym  # noqa: F401
    _ISAACGYM_AVAILABLE = True
except ModuleNotFoundError:
    _ISAACGYM_AVAILABLE = False

print("_ISAACGYM_AVAILABLE is", _ISAACGYM_AVAILABLE)
if _ISAACGYM_AVAILABLE:
    try:
        from .manip_loco.manip_loco import ManipLoco  # noqa: F401
        from .manip_loco.b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO  # noqa: F401
        from .manip_loco.b2z1_config import B2Z1RoughCfg, B2Z1RoughCfgPPO  # noqa: F401
        from legged_gym.utils.task_registry import task_registry

        task_registry.register(
            "b1z1",
            ManipLoco,
            B1Z1RoughCfg(),
            B1Z1RoughCfgPPO(),
            "manip_loco",
        )
        task_registry.register(
            "b2z1",
            ManipLoco,
            B2Z1RoughCfg(),
            B2Z1RoughCfgPPO(),
            "manip_loco",
        )
    except Exception as exc:
        print(f"[legged_gym.envs] Warning: legacy IsaacGym task registration skipped: {exc}")

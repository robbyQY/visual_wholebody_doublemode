class TaskRegistry:
    def make_env(self, *args, **kwargs):
        raise RuntimeError("Use scripts/play_isaaclab.py or IsaacLab gymnasium registration for DirectRLEnv tasks.")

task_registry = TaskRegistry()

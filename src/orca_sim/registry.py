from __future__ import annotations

import gymnasium as gym


def register_envs() -> None:
    registry = gym.registry
    specs = {
        "OrcaHandLeft-v0": "orca_sim.envs:OrcaHandLeft",
        "OrcaHandRight-v0": "orca_sim.envs:OrcaHandRight",
        "OrcaHandCombined-v0": "orca_sim.envs:OrcaHandCombined",
    }
    for env_id, entry_point in specs.items():
        if env_id not in registry:
            gym.register(id=env_id, entry_point=entry_point)

from gym.envs.registration import register

register(
    id="AddictionEnv-v0",
    entry_point="addiction_simulation.envs:AddictionEnv",
)
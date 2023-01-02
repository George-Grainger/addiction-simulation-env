from gym.envs.registration import register

register(
    id="AddictionEnv",
    entry_point="addiction_simulation.envs:AddictionEnv",
)
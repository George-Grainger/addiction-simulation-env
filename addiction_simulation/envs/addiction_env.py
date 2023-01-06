import gym
import numpy as np
from gym import spaces

class AddictionEnv(gym.Env):
    """Class representing the addiction learning environment."""

    metadata = {"render_modes": [None]}
    NUM_STATES = 1
    NUM_ACTIONS = 2

    addictive_actions = {1}

    def __init__(self, rewards: list=[(5, 0.02), (2, 0.02)], render_mode: str=None):
        self.observation_space = spaces.Discrete(AddictionEnv.NUM_STATES)
        self.action_space = spaces.Discrete(AddictionEnv.NUM_ACTIONS)
        self.current_state = self.np_random.integers(self.NUM_STATES, dtype=np.int64)

        self.rewards = rewards

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @property
    def rewards(self) -> list:
        return self._rewards
    
    @rewards.setter
    def rewards(self, val):
        if(type(val) != list):
            raise TypeError("Rewards must be a list of tuples, containing, mean, standard deviation pairs")
        if(len(val) != self.action_space.n):
            raise IndexError("Must be a key value pair for every action")
        if(any(type(el) is not tuple or len(el) != 2 for el in val)):
            raise TypeError("Reward items must be tuple in the from (mean, std)")
        self._rewards = val

    @property
    def current_state(self):
        return self._current_state
    
    @current_state.setter
    def current_state(self, val: int):
        if(val not in range(AddictionEnv.NUM_STATES)):
            raise ValueError(f"State must be integer in range 0 to {AddictionEnv.NUM_STATES - 1}")
        self._current_state = val

    def _get_reward(self, action: int):
        return self.np_random.normal(*self._rewards[action])

    def _increment_state(self, action: int):
        next_state = 0
        self.current_state = next_state
        return next_state

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_state = self.np_random.integers(self.NUM_STATES, dtype=np.int64)
        observation = self.current_state
        info = self._get_info()

        return observation, info

    def step(self, action):   
        assert self.action_space.contains(action), "Invalid Action"

        reward = self._get_reward(action)
        observation = self._increment_state(action)
        info = self._get_info()

        return observation, reward, False, False, info

    def render(self):
        pass

    def close(self):
        pass
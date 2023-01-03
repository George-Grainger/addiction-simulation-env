import gym
from gym import spaces

class AddictionEnv(gym.Env):
    """Class representing the addiction learning environment."""

    metadata = {"render_modes": [None]}
    NUM_STATES = 3

    def __init__(self, render_mode: str=None):
        self.observation_space = spaces.Discrete(AddictionEnv.NUM_STATES)
        self.action_space = spaces.Discrete(4)
        self.iterations = 0
        self.current_state = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @property
    def current_state(self):
        return self._current_state
    
    @current_state.setter
    def current_state(self, val: int):
        if(val not in range(AddictionEnv.NUM_STATES)):
            raise ValueError("Iterations must be greater than 0")
        self._current_state = val

    @property 
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, val: int):
        if(val < 0):
            raise ValueError("Number of iterations must be greater than 0")
        self._iterations = val

    def _get_reward(self, action: int):
        rewards = {
            0: [-5, 0, 5, 10],
            1: [-5, 0, 0, -10],
            2: [-5, -1, -1, -2]
        }
        return rewards.get(self.current_state)[action]

    def _increment_state(self, action: int):
        next_states = {
            0: [1, 1, 0, 1],
            1: [1, 1, 1, 2],
            2: [0, 2, 2, 2]
        }
        next_state = next_states.get(self.current_state)[action]
        self.current_state = next_state
        return next_state

    def _get_info(self):
        return {"distance": 0}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.iterations = 0
        self.current_state = 0
        observation = self.current_state
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = self._get_reward(action)
        observation = self._increment_state(action)
        info = self._get_info()

        self.iterations += 1
        terminated = self.iterations >= 1000

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
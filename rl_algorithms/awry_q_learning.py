from rl_algorithms.q_learning import QLearning
import numpy as np


class AwryQLearning(QLearning):
    """Class to contain Q-table and all parameters with methods to update Q-table and get actions."""

    def __init__(self, n_states: int, n_actions: int, dopamine_surge: float=15, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self._dopamine_surge = dopamine_surge

    @property
    def dopamine_surge(self) -> float:
        return self._dopamine_surge
        
    def _get_td_diff(self, state: int, action: int, reward: float, next_state: int):
        td_diff = reward + (self.gamma * np.max(self.q_table[next_state])) - self.q_table[state, action]
        if(action):
            td_diff = max(td_diff + self.dopamine_surge, self.dopamine_surge)
        return td_diff
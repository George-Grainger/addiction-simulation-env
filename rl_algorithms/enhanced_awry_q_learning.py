import numpy as np
from rl_algorithms.awry_q_learning import AwryQLearning

class EnhancedAwryQLearning(AwryQLearning):
    """Class to contain Q-table and all parameters with methods to update Q-table and get actions."""

    def __init__(self, n_states: int, n_actions: int, sigma: float, lambda_: float, basal_lim: float, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self._sigma = sigma
        self._lambda = lambda_
        self._basal_lim = basal_lim
        self._action_avg = 0
        self._bias = 0

    @property
    def sigma(self) -> float:
        return self._sigma

    def exponential_avg(self, influence: float, prev: float, to_add: float):
        return (1-influence)*prev + influence*to_add
        
    def _get_td_diff(self, state: int, action: int, reward: int, next_state: int):
        next_action = self.get_action(next_state)
        td_diff = reward + self.q_table[next_state, next_action] - self.q_table[state, action]
          
        if(action):
            bl = self._basal_lim
            td_diff = max(td_diff + (self.dopamine_surge - self._bias), (self.dopamine_surge - self._bias))
            for_avg = td_diff - self.q_table[next_state, next_action] + self.q_table[state, action]
        else:
            bl = 0
            for_avg = reward
        
        td_diff -= self._action_avg     
        self._action_avg = self.exponential_avg(self.sigma, self._action_avg, for_avg)
        self._bias = self.exponential_avg(self._lambda, self._bias, bl)

        return td_diff

    def wipe(self):
        super().wipe()
        self._bias = 0
        self._action_avg = 0

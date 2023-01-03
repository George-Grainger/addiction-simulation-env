from rl_algorithms.q_learning import QLearning


class AwryQLearning(QLearning):
    """Class to contain Q-table and all parameters with methods to update Q-table and get actions."""

    def __init__(self, dopamine_surge: int, n_states: int, n_actions: int, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self._dopamine_surge = dopamine_surge

    @property
    def dopamine_surge(self) -> float:
        return self._dopamine_surge
        
    def _get_td_diff(self, obs_i: int, action: int, reward: int, next_obs_i: int):
        baseline = super()._get_td_diff(obs_i, action, reward, next_obs_i)
        return max(baseline, self.dopamine_surge)
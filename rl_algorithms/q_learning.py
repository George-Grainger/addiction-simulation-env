import os, pickle
import numpy as np

from rl_algorithms.rl_base import RLBase


class QLearning(RLBase):
    """Class to contain Q-table and all parameters with methods to update Q-table and get actions."""

    def __init__(self, n_states: int, n_actions: int, epsilon_max: float=1.0, epsilon_min: float=0.01, gamma: float=0.99, saved_path: str=None, **kwargs):
        """
        function to initalise the QLearning class
        n_states is the number of discrete (discretised if continuous) states in the environment
        n_actions is the number of discrete actions (q learning will only perform with discrete action space)
        gamma is a float which is the discount factor of future rewards
        epsilon max is the maximum exploration rate of the agent
        epsilon min is the minimum exploration rate of the agent
        lr is the learning rate of the agent
        saved_path 
        """
        super().__init__(n_actions, **kwargs)
        self.epsilon = epsilon_max
        self._epsilon_max = epsilon_max 
        self._epsilon_min = epsilon_min
        self.gamma = gamma
        self._q_table = np.zeros((n_states, n_actions))

        #load a saved model (q-table) if provided
        if saved_path:
            if os.path.isfile(saved_path):
                with open(saved_path, "rb") as handle:
                    self._q_table = pickle.load(handle)
            else:
                raise FileNotFoundError

    #-------------------------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, val: float):
        if val < 0 or val > 1: 
            raise ValueError("epsilon (exploration probability) must have a value between 0 and 1 (inclusive).")
        
        if not isinstance(val, float):
            raise TypeError("epsilon (exploration probability) must be a float")

        self._epsilon = val
    
    @property
    def epsilon_max(self) -> float:
        return self._epsilon_max

    @property
    def epsilon_min(self) -> float:
        return self._epsilon_min

    @property
    def q_table(self):
        return self._q_table

    #-------------------------------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------------------------------

    def _get_td_diff(self, state: int, action: int, reward: float, next_state: int):
        return reward + (self.gamma * np.max(self.q_table[next_state])) - self.q_table[state, action]
        
    def save_model(self, path: str):
        """
        function to save the model (q-table) to a file
        path is a string of the path to the file where the model will be saved
        """
        with open(path, "wb") as handle:
            pickle.dump(self.q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_action(self, state: int):
        """
        function to get the action based on the current observation using an epsilon-greedy policy
        state is the current observation of the state indexed for the q_table (done using index_obv function)
        returns the action to take as an int
        Note: index_obv function should be done outside of this class to prevent performance issues
        """
        #take random action with probability epsilon (explore rate)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            #policy is greedy
            action = np.argmax(self.q_table[state])

        return action

    def decay_epsilon(self, n_t: int):
        """
        function to reduce value of learning parameters decaying exponentially at the rate of the decay property
        n_t is the current episode number
        """
        self.epsilon *= self.epsilon_decay ** (n_t + 1)

    def train(self, state: int, action: int, reward: float, next_state: int):
        """
        function to train agent by applying the q-value update rule to the q-table
        state is the observation from the environment indexed for the q_table
        action is the action taken by the agent
        reward is the reward provided by the environment after taking action in current state
        next_state is the observation after taking action in the current state indexed for the q_table
        Note: index_obv function should be done outside of this class to prevent performance issues
        """
        super().train(state, action, reward, next_state)
        td_diff = self._get_td_diff(state, action, reward, next_state)
        self.q_table[state, action] += self.lr * td_diff

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)
    
    def wipe(self):
        self._q_table = np.zeros_like(self.q_table)
        self.epsilon = self.epsilon_max
        self.lr = self._init_lr
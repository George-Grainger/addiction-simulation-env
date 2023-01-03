import os, pickle
import numpy as np

from rl_algorithms.rl_base import RLBase


class QLearning(RLBase):
    """Class to contain Q-table and all parameters with methods to update Q-table and get actions."""

    def __init__(self, n_states: int, n_actions: int, gamma: float=0.99, epsilon_max: float=1.0, epsilon_min: float=0.01, lr: float=0.7, decay: float=0.999, saved_path: str=None, **kwargs):
        """
        function to initalise the QLearning class
        n_states is the number of discrete (discretised if continuous) states in the environment
        n_actions is the number of discrete actions (q learning will only perform with discrete action space)
        gamma is a float which is the discount factor of future rewards
        epsilon max is the maximum exploration rate of the agent
        epsilon min is the minimum exploration rate of the agent
        lr is the learning rate of the agent
        lr_decay is the rate at which the learning rate will decay exponentially
        saved_path 
        """
        super().__init__(**kwargs)

        self.gamma = gamma
        self.lr = lr
        self.decay = decay
        self.n_actions = n_actions
        self.epsilon = epsilon_max

        self._epsilon_max = epsilon_max 
        self._epsilon_min = epsilon_min
        self._q_table = np.zeros((n_states, self.n_actions))

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

    def _get_td_diff(self, obs_i: int, action: int, reward: int, next_obs_i: int):
        td_target = reward + (self.gamma * np.max(self.q_table[next_obs_i]))
        return td_target - self.q_table[obs_i, action]

    def save_model(self, path: str):
        """
        function to save the model (q-table) to a file
        path is a string of the path to the file where the model will be saved
        """
        with open(path, "wb") as handle:
            pickle.dump(self.q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_action(self, obs_i: int):
        """
        function to get the action based on the current observation using an epsilon-greedy policy
        obs_i is the current observation of the state indexed for the q_table (done using index_obv function)
        returns the action to take as an int
        Note: index_obv function should be done outside of this class to prevent performance issues
        """
        #take random action with probability epsilon (explore rate)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            #policy is greedy
            action = np.argmax(self.q_table[obs_i])

        return action

    def update_parameters(self, n_t: int):
        """
        function to reduce value of learning parameters decaying exponentially at the rate of the decay property
        n_t is the current episode number
        """
        self.epsilon *= self.decay ** (n_t + 1)
        self.lr *= self.decay ** (n_t + 1)

    def train(self, obs_i: int, action: int, reward: int, next_obs_i: int):
        """
        function to train agent by applying the q-value update rule to the q-table
        obs_i is the observation from the environment indexed for the q_table (done using index_obv function)
        action is the action taken by the agent
        reward is the reward provided by the environment after taking action in current state
        next_obs_i is the observation after taking action in the current state indexed for the q_table (done using the index_obv function)
        Note: index_obv function should be done outside of this class to prevent performance issues
        """
        super().train(obs_i, action, reward, next_obs_i)
        td_diff = self._get_td_diff(obs_i, action, reward, next_obs_i)
        self.q_table[obs_i, action] += self.lr * td_diff
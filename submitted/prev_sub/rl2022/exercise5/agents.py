from abc import ABC, abstractmethod
from collections import defaultdict
import random
import numpy as np
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        for q_table, n_acts in zip(self.q_tables, self.n_acts):
            act_vals = [q_table[(a)] for a in range(n_acts)]
            max_val = max(act_vals)
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
            if random.random() < self.epsilon:
                actions.append(random.randint(0, n_acts - 1))
            else:
                actions.append(random.choice(max_acts))
        return actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        for i, (q_table, action, reward, done, n_acts) in enumerate(zip(self.q_tables, actions, rewards, dones, self.n_acts)):
            max_q = max([q_table[(a)] for a in range(n_acts)]) 
            q_table[(action)] += self.learning_rate * ((reward + self.gamma * (1 - done) * max_q) - q_table[(action)])
            self.q_tables[i] = q_table
            updated_values.append(q_table[(action)])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        pass


class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            q_table = self.q_tables[i]
            ev = 1
            if self.models[i] == 0:
                ev = 0
            if random.random() < self.epsilon or ev == 0:
                joint_action.append(random.randint(0, self.n_acts[i] - 1))
            else:
                n_acts = self.n_acts[i]
                n_acts_opp = self.n_acts[(i + 1) % 2]
                evs = []
                for action in range(n_acts):
                    ev_state_action = 0
                    sum_of_opp = 0
                    for action_opp in range(n_acts_opp):
                        sum_of_opp += self.models[i][action_opp]
                    for action_opp in range(n_acts_opp):
                        ev_state_action += (self.models[i][action_opp] / max(1,sum_of_opp)) * q_table[(action, action_opp)]
                        #  dividing its count by the sum over the counts of all possible joint actions of the other agents a′−i.
                    evs.append(ev_state_action)
                max_ev = max(evs)
                max_acts = [idx for idx, act_ev in enumerate(evs) if act_ev == max_ev]
                joint_action.append(random.choice(max_acts))
        return joint_action

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###n
        for i in range(self.num_agents):
            q_table = self.q_tables[i]
            reward = rewards[i]
            done = dones[i]
            opp = (i + 1) % 2
            action_opp = actions[opp]
            self.models[i][action_opp] += 1 if self.models[i][action_opp] else 1
            q_value_old = q_table[tuple(actions)]
            evs = []
            # sum_of_opp=0
            for action_next in range(self.n_acts[i]): # for each action for the agent i
                ev_state_actionNext = 0
                sum_of_opp=0
                for action_next_opp in range(self.n_acts[opp]): # for each action of the opponent opp
                    sum_of_opp += self.models[i][action_opp]
                for action_next_opp in range(self.n_acts[opp]):
                    ev_state_actionNext += (self.models[i][action_next_opp] / max(1,sum_of_opp)) * q_table[(action_next,action_next_opp)]
                evs.append(ev_state_actionNext)
            next_best_ev = max(evs) if not done else 0
            q_table[tuple(actions)] = q_value_old + self.learning_rate * ((reward + self.gamma * next_best_ev) - q_value_old)
            self.q_tables[i] = q_table
            updated_values.append(q_table[tuple(actions)])
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        pass


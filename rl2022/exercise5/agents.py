from abc import ABC, abstractmethod
from collections import defaultdict
import random
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
            act_vals = [q_table[a] for a in range(n_acts)]
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
            old_q = q_table[action]
            max_q = max([q_table[a] for a in range(n_acts)]) 
            q_table[action] = old_q + self.learning_rate * (reward + self.gamma * (1 - done) * max_q - old_q)
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
        self.epsilon = 1.0 - (min(1.0, timestep / (0.07 * max_timestep))) * 0.95


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
#         for i in range(self.num_agents):
#             q_table = self.q_tables[i]
#             ev = 1
#             if self.c_obss[i][obs] == 0:
#                 ev = 0
#             if random.random() < self.epsilon or ev == 0:
#                 joint_action.append(random.randint(0, self.n_acts[i] - 1))
#             else:
#                 j = (i + 1) % 2
#                 n_acts = self.n_acts[i]
#                 n_acts_opp = self.n_acts[j]
#                 evs = []
#                 for action in range(n_acts):
#                     ev_state_action = 0
#                     for action_opp in range(n_acts_opp):
#                         ev_state_action += (self.models[i][obs][action_opp] / self.c_obss[i][obs]) * q_table[obs,(
#                             action, action_opp)]
#                     evs.append(ev_state_action)
#                 max_ev = max(evs)
#                 max_acts = [idx for idx, act_ev in enumerate(evs) if act_ev == max_ev]
#                 joint_action.append(random.choice(max_acts))
# # 

#         for current_agent in range(self.num_agents):
#             opp = (current_agent + 1) % 2
#             c_obss_agent = self.models[opp]
#             q_tab = self.q_tables[current_agent]
#             model = self.models[current_agent]

#             if random.random() < self.epsilon or c_obss_agent[0] == 0:
#                 joint_action.append(random.randint(0, self.n_acts[current_agent] - 1))
#             else:
#                 values = []
#                 for own_action in range(self.action_spaces[0].n):
#                     expectation_vals = 0
#                     for others_action in range(self.action_spaces[0].n):
#                         action_key = (own_action, others_action)
#                         action_key_alt = (others_action, own_action)
                        
#                         # if c_obss_agent[obss[current_agent]] != 0:
#                         if current_agent == 0: #own turn
#                             expectation_vals += model[others_action] * q_tab[(action_key)]
#                         else:
#                             expectation_vals += model[others_action] * q_tab[(action_key_alt)]
#                         # else:
#                             # expectation_vals = 0
#                     values.append(expectation_vals)
#                 action = max(values)
#                 joint_action.append(action)
        # return joint_action
# 
        # joint_action = []
        # # print(self.models)
        # ### PUT YOUR CODE HERE ###
        # for i in range(self.num_agents):
        #     opp = (i + 1) % 2
        #     EV = []
        #     if random.random() < self.epsilon:
        #         joint_action.append(random.randint(0, self.n_acts[i] - 1))
        #     else:
        #         if self.c_obss[i][obss[i]] == 0.0:
        #             joint_action.append(random.randint(0, self.n_acts[i] - 1))
        #         else:
        #             if i == 0: # own turn
        #                 for ai in range(self.n_acts[i]):
        #                     # count
        #                     # * Q which is q_table( where agent i applies action ai, and opponent[all other agents] apply a_i )
        #                     EV.append(sum([self.models[i][a_i] / self.models[i][a_i] * self.q_tables[i][
        #                         (ai, a_i)] for a_i in range(self.n_acts[i])]))
        #                         # ai: own, a_i: opponent
        #             else: # opponent's turn
        #                 for ai in range(self.n_acts[i]):
        #                     EV.append(sum([self.models[i][a_i] / self.models[i][ai] * self.q_tables[i][
        #                         (a_i, ai)] for a_i in range(self.n_acts[i])]))
        #             joint_action.append(max(EV))


        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        # self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

        for i in range(self.num_agents):
            q_table = self.q_tables[i]
            ev = 1
            # if self.c_obss[i][obs] == 0:
            # print(i,self.models[0][0], self.models[0], self.models) # self.models = {list of => {0: 173, 1: 169, 2: 163}, {} }
            # if i == 1:
                # vbnm,
            if self.models[i] == 0:
                # print(self.models)
                ev = 0
            if random.random() < self.epsilon or ev == 0:
                joint_action.append(random.randint(0, self.n_acts[i] - 1))
            else:
                # j = (i + 1) % 2
                n_acts = self.n_acts[i]
                n_acts_opp = self.n_acts[(i + 1) % 2]
                evs = []
                for action in range(n_acts):
                    ev_state_action = 0
                    sum_of_opp = 0
                    for action_opp in range(n_acts_opp):
                        sum_of_opp += self.models[i][action_opp]
                    for action_opp in range(n_acts_opp):
                        ev_state_action += (self.models[i][action_opp] / sum_of_opp) * q_table[(action, action_opp)]
                        #  dividing its count by the sum over the counts of all possible joint actions of the other agents a′−i.
                    evs.append(ev_state_action)
                max_ev = max(evs)
                max_acts = [idx for idx, act_ev in enumerate(evs) if act_ev == max_ev]
                joint_action.append(random.choice(max_acts))


        return joint_action
        # for i in range(self.num_agents):
        #     EV = []
        #     if random.random() < self.epsilon:
        #         joint_action.append(random.randint(0, self.n_acts[i] - 1))
        #     else:
        #         # if self.c_obss[i][obss[i]] == 0.0:
        #         # joint_action.append(random.randint(0, self.n_acts[i] - 1))
        #         # else:
        #         if i == 0:
        #             for ai in range(self.n_acts[i]):
        #                 # models: count: Ca-i
        #                 EV.append(sum([self.models[i][a_i] * self.q_tables[i][
        #                     ((ai, a_i))] for a_i in range(self.n_acts[i])]))
        #         else:
        #             for ai in range(self.n_acts[i]):
        #                 EV.append(sum([self.models[i][a_i]  * self.q_tables[i][
        #                     ((a_i, ai))] for a_i in range(self.n_acts[i])]))
        #         joint_action.append(random.choice(EV))
        # return joint_action

        # for num_agent, (q_table, n_acts) in enumerate(zip(self.q_tables, self.n_acts)):
        #     ev = 1
        #     if num_agent == 0:
        #         ev = 0
        #     if random.random() < self.epsilon or ev == 0:
        #         joint_action.append(random.randint(0, n_acts - 1))
        #     else:
        #         j = (num_agent + 1) % 2 # if 0 then 1, if 1 then 0 => opponent
        #         n_acts_opp = self.n_acts[j]
        #         evs = []
        #         for action in range(n_acts):
        #             ev_state_action = 0
        #             for action_opp in range(n_acts_opp):
        #                 ev_state_action += (self.models[num_agent][action_opp] ) * q_table[(action, action_opp)]
        #             evs.append(ev_state_action)
        #         max_ev = max(evs)
        #         max_acts = [idx for idx, act_ev in enumerate(evs) if act_ev == max_ev]
        #         joint_action.append(random.choice(max_acts))

        # return joint_action

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
        ### PUT YOUR CODE HERE ###
        for i in range(self.num_agents):
            # obs = obss[i]
            reward = rewards[i]
            # n_obs = n_obss[i]
            done = dones[i]
            q_table = self.q_tables[i]
            j = (i + 1) % 2
            action_opp = actions[j]
            # self.c_obss[i][obs] += 1 if self.c_obss[i][obs] else 1
            # self.models[i][obs][action_opp] += 1 if self.models[i][obs][action_opp] else 1
            self.models[i][action_opp] += 1 if self.models[i][action_opp] else 1
            q_value_old = q_table[tuple(actions)]
            evs = []
            # sum_of_opp=0
            for action_next in range(self.n_acts[i]):
                ev_state_actionNext = 0
                sum_of_opp=0
                for action_next_opp in range(self.n_acts[j]):
                    sum_of_opp += self.models[i][action_opp]
                # print(sum_of_opp)
                for action_next_opp in range(self.n_acts[j]):
                    # (self.models[i][action_opp] / sum(self.models[i]))
                    # ev_state_actionNext += (self.models[i][n_obs][action_next_opp] / self.c_obss[i][n_obs]) * q_table[n_obs,(action_next,action_next_opp)]
                    ev_state_actionNext += (self.models[i][action_next_opp] / sum_of_opp) * q_table[(action_next,action_next_opp)]
                
                evs.append(ev_state_actionNext)
            next_best_ev = max(evs) if not done else 0
            q_table[tuple(actions)] = q_value_old + self.learning_rate * (reward + self.gamma * next_best_ev - q_value_old)
            self.q_tables[i] = q_table
            updated_values.append(q_table[tuple(actions)])
        return updated_values
        # for current_agent in range(self.num_agents):
        #     current_agent_action = actions[current_agent]
        #     other_agent_action = actions[1-current_agent]

        #     q_tab = self.q_tables[current_agent]
        #     model = self.models[current_agent]
        #     # c_obss_agent = self.c_obss[current_agent]

        #     action_key = tuple(actions)
        #     model[other_agent_action] += 1
        #     # c_obss_agent[obss[current_agent]] +=1

        #     joint_action = []
        #     for own_action in range(3):
        #         expectation_vals = 0 
        #         for others_action in range(3):
        #             #print(f"action_key: {action_key}, q_tab value: {q_tab[(0,action_key)]}, N(s): {c_obss_agent[0]}, model: {model[0][others_action]}")
        #             action_key_own = (own_action, others_action)
        #             action_key_alt = (others_action, own_action)
                    
        #             # if c_obss_agent[n_obss[current_agent]] != 0: # Avoid the divide by zero error in the tests
        #             if current_agent == 0: #own agents turn
        #                 expectation_vals += model[others_action] * q_tab[(action_key_own)]
        #             else: #other agents turn
        #                 expectation_vals += model[others_action] *q_tab[(action_key_alt)]
        #             # else:
        #                 # expectation_vals = 0

        #         joint_action.append(expectation_vals)

        #     action_ = max(joint_action)
        #     # float -> int works
        #     update = float(q_tab[(action_key)] + self.learning_rate * (rewards[current_agent] + self.gamma * action_ - q_tab[(action_key)]))
        #     q_tab[(action_key)] = update
        #     updated_values.append(update)

        # for i in range(self.num_agents):
        #     # obs = obss[i]
        #     reward = rewards[i]
        #     # n_obs = n_obss[i]
        #     done = dones[i]
        #     q_table = self.q_tables[i]
        #     j = (i + 1) % 2
        #     action_opp = actions[j]
        #     # self.c_obss[i][obs] += 1 if self.c_obss[i][obs] else 1
        #     self.models[i][action_opp] += 1 if self.models[i][action_opp] else 1
        #     q_value_old = q_table[tuple(actions)]
        #     evs = []
        #     for action_next in range(self.n_acts[i]):
        #         ev_state_actionNext = 0
        #         for action_next_opp in range(self.n_acts[j]):
        #             ev_state_actionNext += (self.models[i][action_next_opp] ) * q_table[(action_next,action_next_opp)]
        #         evs.append(ev_state_actionNext)
        #     next_best_ev = max(evs) if not done else 0
        #     q_table[tuple(actions)] = q_value_old + self.learning_rate * (reward + self.gamma * next_best_ev - q_value_old)
        #     self.q_tables[i] = q_table
        #     updated_values.append(q_table[tuple(actions)])
        return [] #updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q5")

from collections import defaultdict
import random
import numpy as np

class TrustAwareAgent():
    ACTIONS = ["NO_ERROR", "ERROR"]
    
    def __init__(self):
        self.identity = "TRUST_AWARE"
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.trust_alpha = 0.01
        # Q-table: keys = actual_value, values = list of Q-values for actions
        self.Q = defaultdict(lambda: [0.0 for _ in self.ACTIONS])

        self.trust_reinforcer_1 = 0.5
        self.trust_reinforcer_2 = 0.5
    
    def internal_moral_reward(self, state, action, next_state):
        if state['actual_value'] == "ERROR":
            if action == 'ERROR':
                return 1
            else:
                return -1
        elif state['actual_value'] == "NO_ERROR":
            if action == 'NO_ERROR':
                return 1
            else:
                return -1
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        else:
            q_values = self.Q[state['actual_value']]
            max_index = q_values.index(max(q_values))
            return self.ACTIONS[max_index]
    
    def learn(self, state, action, reward, next_state, reinforcer_1, reinforcer_2):
        action_index = self.ACTIONS.index(action)
        next_max = max(self.Q[next_state['actual_value']])

        if reinforcer_1:
            self.Q[state['actual_value']][action_index] += self.alpha * (
                self.trust_reinforcer_1 * reward + self.gamma * next_max -
                self.Q[state['actual_value']][action_index]
            )
        if reinforcer_2:
            self.Q[state['actual_value']][action_index] += self.alpha * (
                self.trust_reinforcer_2 * reward + self.gamma * next_max - 
                self.Q[state['actual_value']][action_index]
            )

    def meta_learn(self, cumulative_reward, internal_cumulative_reward, reinforcer_1, reinforcer_2):
        if reinforcer_1:
            self.trust_reinforcer_1 = max(0, min(1, 
            self.trust_reinforcer_1 + 
            self.trust_alpha * np.sign(cumulative_reward * internal_cumulative_reward)))
        if reinforcer_2:
            self.trust_reinforcer_2 = max(0, min(1, 
            self.trust_reinforcer_2 + 
            self.trust_alpha * np.sign(cumulative_reward * internal_cumulative_reward)))

       
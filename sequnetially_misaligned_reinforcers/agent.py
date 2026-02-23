from collections import defaultdict
import random

class Agent():
    ACTIONS = ["NO_ERROR", "ERROR"]
    
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        # Q-table: keys = actual_value, values = list of Q-values for actions
        self.Q = defaultdict(lambda: [0.0 for _ in self.ACTIONS])
    
    def internal_moral_reward(self, state, action, next_state):
        if state['actual_value'] == "ERROR":
            if action == 'ERROR':
                return 1
            else:
                return -100
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
    
    def learn(self, state, action, reward, next_state):
        action_index = self.ACTIONS.index(action)
        next_max = max(self.Q[next_state['actual_value']])
        self.Q[state['actual_value']][action_index] += self.alpha * (
            reward + self.gamma * next_max - self.Q[state['actual_value']][action_index]
        )
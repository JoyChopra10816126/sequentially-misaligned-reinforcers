class Reinforcer_1():
    def __init__(self):
        pass
    
    def reward(self, state, action, next_state):
        return state['actual_value'] == action

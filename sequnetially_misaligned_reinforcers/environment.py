import random

class Environment():
    
    def __init__(self):
        # Online stream
        self.state = {
            'sequence_number': 1,
            'actual_value': self.generate_actual_value()
        }

    @staticmethod
    def generate_actual_value():
        # Monitoring system
        ENVIRONMENT_VALUES = ["NO_ERROR", "ERROR"]
        return random.choice(ENVIRONMENT_VALUES)
    
    def step(self, action):
        # Externally controlled environment
        self.state = {
            'sequence_number': self.state['sequence_number'] + 1,
            'actual_value': self.generate_actual_value()
        }
        return self.state
         
    def reset(self):
        self.state = {
            'sequence_number': 1,
            'actual_value': self.generate_actual_value()
        }
        return self.state
    
    def render(self):
        print(self.state)
    
from environment import Environment
from agent import Agent
from reinforcer_1 import Reinforcer_1
from reinforcer_2 import Reinforcer_2

environment = Environment()
agent = Agent()
reinforcer_1 = Reinforcer_1()
reinforcer_2 = Reinforcer_2()

cumulative_reward = 0
internal_cumulative_reward_1 = 0
internal_cumulative_reward_2 = 0

state = environment.reset()
for step in range(10000):
    action = agent.act(state)
    next_state = environment.step(action)
    reward = reinforcer_1.reward(state, action, next_state)
    cumulative_reward += reward
    internal_cumulative_reward_1 += agent.internal_moral_reward(state, action, next_state)
    agent.learn(state, action, reward, next_state)
    state = next_state


print("Reinforcer 1: Reward:", cumulative_reward, internal_cumulative_reward_1)

cumulative_reward = 0

state = environment.reset()
for step in range(10000):
    action = agent.act(state)
    next_state = environment.step(action)
    reward = reinforcer_2.reward(state, action, next_state)
    internal_cumulative_reward_2 += agent.internal_moral_reward(state, action, next_state)
    cumulative_reward += reward
    agent.learn(state, action, reward, next_state)
    state = next_state


print("Reinforcer 2: Reward:", cumulative_reward, internal_cumulative_reward_2)


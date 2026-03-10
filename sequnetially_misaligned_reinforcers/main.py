from environment import Environment
from naive_agent import NaiveAgent
from trust_aware_agent import TrustAwareAgent
from reinforcer_1 import Reinforcer_1
from reinforcer_2 import Reinforcer_2

environment = Environment()
agents = [NaiveAgent(), TrustAwareAgent()]

reinforcer_1 = Reinforcer_1()
reinforcer_2 = Reinforcer_2()


for agent in agents:
    print(agent.identity)
    cumulative_reward = 0
    internal_cumulative_reward_1 = 0


    for step in range(100):
        state = environment.reset()
        cumulative_reward = 0
        internal_cumulative_reward_1 = 0
        for inner_step in range(100):
            action = agent.act(state)
            next_state = environment.step(action)
            reward = reinforcer_1.reward(state, action, next_state)
            cumulative_reward += agent.trust_reinforcer_1 * reward
            internal_reward = agent.internal_moral_reward(state, action, next_state)
            internal_cumulative_reward_1 += agent.trust_reinforcer_1 * internal_reward
            agent.learn(state, action, reward, next_state, True, False)
            state = next_state
    
        agent.meta_learn(cumulative_reward, internal_cumulative_reward_1, True, False)

    cumulative_reward = 0
    internal_cumulative_reward_1 = 0

    for step in range(1000):
        action = agent.act(state)
        next_state = environment.step(action)
        reward = reinforcer_1.reward(state, action, next_state)
        cumulative_reward += agent.trust_reinforcer_1 * reward
        internal_reward = agent.internal_moral_reward(state, action, next_state)
        internal_cumulative_reward_1 += agent.trust_reinforcer_1 * internal_reward
        agent.learn(state, action, reward, next_state, True, False)
        state = next_state

    print("Reinforcer 1: Reward:", agent.trust_reinforcer_1, cumulative_reward, internal_cumulative_reward_1)

    for step in range(100):
        state = environment.reset()
        cumulative_reward = 0
        internal_cumulative_reward_2 = 0
        for inner_step in range(100):
            action = agent.act(state)
            next_state = environment.step(action)
            reward = reinforcer_2.reward(state, action, next_state)
            cumulative_reward += agent.trust_reinforcer_2 * reward
            internal_cumulative_reward_2 += agent.trust_reinforcer_2 * agent.internal_moral_reward(state, action, next_state)
            agent.learn(state, action, reward, next_state, False, True)
            state = next_state
        
        agent.meta_learn(cumulative_reward, internal_cumulative_reward_2, False, True)

    cumulative_reward = 0
    internal_cumulative_reward_2 = 0

    for step in range(1000):
        action = agent.act(state)
        next_state = environment.step(action)
        reward = reinforcer_2.reward(state, action, next_state)
        internal_cumulative_reward_2 += agent.internal_moral_reward(state, action, next_state)
        cumulative_reward += agent.trust_reinforcer_2 * reward
        agent.learn(state, action, reward, next_state, False, True)
        state = next_state

    print("Reinforcer 2: Reward:", agent.trust_reinforcer_2, cumulative_reward, internal_cumulative_reward_2)


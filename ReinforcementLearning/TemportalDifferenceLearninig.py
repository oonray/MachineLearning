import gym

games = 100
env = gym.make("CartPole-v1")

State = []
Last_Action = []
Actions =  env.action_space.n

Time = 0

LearningRate = 1e-3

Discount = 0.8

State.append(env.reset())

for _ in range(games):
    State_T = State[Time]
    env.

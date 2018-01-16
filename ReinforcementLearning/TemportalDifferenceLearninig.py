import gym

games = 100
env = gym.make("CartPole-v1")

State = []
Last_Action = []
Actions = env.action_space.n

Time = 0

LearningRate = 1e-3

Discount = 0.8

State.append(env.reset())

for _ in range(games):
    env.reset()
    Done = False
    State_T = State[Time]
    env.render()
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        Last_Action.append(action)
        State_N = State_T+(LearningRate*((reward+Discount)*State_T-state))
        State.append(State_N)
        Time += 1
        Done = done
        if done:
            break


print([S for S in State])



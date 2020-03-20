import gym
import gym_lqr
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

def act(policy, state):
    state_ = torch.tensor(state).unsqueeze(0)
    with torch.no_grad():
        action = policy(state_)[0][0].numpy()

    return action

env = gym.make('lqr-v0')
policy_net, value_net, running_state = pickle.load(open("learned_models/ppo/2-lqr-v0.p", "rb"))
N = 1000
xn = np.zeros((N,3))
x = env.reset()
for i in range(N):
    xn[i,:] = x
    action = act(policy_net, x)
    x, rewards, dones, info = env.step(action)
env.close()
en = np.linalg.norm(xn,axis=1)
plt.plot(en)
plt.show()

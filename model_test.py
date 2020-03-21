import gym
import gym_lqr
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from core.agent import Agent

env = gym.make('lqr-v0')
policy_net, value_net, running_state = pickle.load(open("learned_models/ppo/3-lqr-v0.p", "rb"))
device = torch.device('cpu')
agent = Agent(env, policy_net, device, running_state=running_state)



N = 1000
xn = np.zeros((N,3))
x = env.reset()
for i in range(N):
    xn[i,:] = x
    action = agent.act(x)
    x, rewards, dones, info = env.step(action)
env.close()
en = np.linalg.norm(xn,axis=1)

plt.style.use('ggplot')
plt.plot(en)
plt.show()

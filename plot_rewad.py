import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
rc('text', usetex=True)
rc('font', family='serif')



with open('logs/ppo/3-lqr-v0.pkl','rb') as f:
    log_ppo = pickle.load(f)

with open('logs/trpo/3-lqr-v0.pkl','rb') as f:
    log_trpo = pickle.load(f)



ppo_samples = log_ppo['total_steps']
ppo_reward  = log_ppo['reward']
trpo_samples = log_trpo['total_steps']
trpo_reward  = log_trpo['reward']

plt.style.use('ggplot')
ax = plt.subplot(111)
plt.plot(ppo_samples, ppo_reward, label='ppo')
plt.plot(trpo_samples, trpo_reward, label='trpo')
plt.xlabel(r'action steps')
plt.ylabel(r'average reward')
plt.legend()
plt.show()

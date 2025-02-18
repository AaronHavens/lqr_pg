import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pickle
rc('text', usetex=True)
rc('font', family='serif')


with open('logs/ppo/1e-lqr-v0.pkl','rb') as f:
    log_ppo = pickle.load(f)

with open('logs/trpo/1e-lqr-v0.pkl','rb') as f:
    log_trpo = pickle.load(f)

ppo_samples = log_ppo['total_steps']
trpo_samples = log_trpo['total_steps']
ppo_reward  = log_ppo['reward']
ppo_reward_eval = log_ppo['reward_eval']
trpo_reward  = log_trpo['reward']
trpo_reward_eval = log_trpo['reward_eval']
lqr = np.ones(np.shape(ppo_samples))*-55.0

plt.style.use('ggplot')
ax = plt.subplot(111)
#plt.plot(ppo_samples, ppo_reward, c='b',label='ppo train')
plt.plot(ppo_samples, ppo_reward_eval, c='b', label='ppo')
#plt.plot(trpo_samples, trpo_reward, label='trpo train')
plt.plot(trpo_samples, trpo_reward_eval, c='r', label='trpo')
plt.plot(trpo_samples, lqr, c='g', linestyle='dashed', label='lqr dare feedback')
plt.xlabel(r'action steps')
plt.ylabel(r'average reward')
plt.legend()
plt.show()

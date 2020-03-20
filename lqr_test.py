import gym
import gym_lqr
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as alg




e = gym.make("lqr-v0")
gamma = 0.98
X = np.matrix(alg.solve_discrete_are(np.sqrt(gamma)*e.A, e.B, e.Q, e.R/gamma))
K = np.matrix(alg.inv(e.B.T*X*e.B+e.R)*(e.B.T*X*e.A))


N = 10000
xn = np.zeros((N,3))
x = e.reset()
max_u = 0
for i in range(N):
    xn[i,:] = x
    u = np.array(np.matmul(-K,x))[0]
    if np.amax(u) > max_u: max_u = np.amax(u)
    x, r, done, _ = e.step(u)
print(max_u)
e.close
en = alg.norm(xn, axis=1)

plt.plot(en)
plt.show()

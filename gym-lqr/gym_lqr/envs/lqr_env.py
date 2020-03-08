import gym
from gym import spaces
import numpy as np


class LQREnv(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        self.Q = np.eye(3)
        self.R = np.eye(2)
        self.W = np.diag([0.1, 0.05, 0.1])
        self.A = np.array([ [0.99, 0.01,0],
                            [0.01, 0.98, 0.01],
                            [0.5,0.12,0.97]]
                            )
        self.B = np.array([ [1, 0.1],
                            [0, 0.1],
                            [0, 0.1]]
                            )
        high = np.array([np.inf, np.inf, np.inf])
        high_u = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Box(low=-high_u, high=high_u)

    def step(self, u):
        x = self.state
        w = np.random.multivariate_normal([0,0,0], cov=self.W)
        x_next = np.matmul(self.A, x) + np.matmul(self.B, u) + w
        r = 0
        r += np.matmul(x.T, np.matmul(self.Q, x)) 
        r += np.matmul(u.T, np.matmul(self.R, u))
        self.state = x_next
        self.t += 1
        return x_next, r , self.isDone(), None

    def reset(self):
        self.t = 0
        x = np.random.uniform(low=-1, high=1, size=(3,))
        self.state = x
        return x

    def render(self, mode='human', close=False):
        return None

    def isDone(self):
        done = False
        if self.t > 200:
            done = True
        return done

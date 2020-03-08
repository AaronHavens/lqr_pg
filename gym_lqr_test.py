import gym
import gym_lqr


e = gym.make('lqr-v0')
e.reset()

for i in range(100):        
    x,r,done,_ = e.step(e.action_space.sample())
    print(x)
    print(r)
    print('-----')


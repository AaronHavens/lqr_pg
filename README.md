# TRPO and PPO + LQR Gym Environment

## Dependencies
* Pytorch
* Gym
### Install LQR Env
```
cd lqr_pg
python -m pip install --user gym-lqr
```
## LQR Gym with Gaussian Noise
$$x_{k+1} = A x_k + B u_k + w_k,\quad w_k \sim \mathcal{N}(0,W)$$

## Example Result
![Image](https://github.com/AaronHavens/lqr_pg/blob/master/figures/ppo_trpo_reward.png?raw=true)


import gym 
import mujoco_py
from gym.envs.mujoco import mujoco_env as mj
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TRPO
from stable_baselines.bench import Monitor
import os
import matplotlib.pyplot as plt
def main():
    train = True
    n_cpu = 16
    env = SubprocVecEnv([lambda: gym.make('tensegrity:TensLeg-v0') for i in range(n_cpu)])
    #env= gym.make('tensegrity:TensLeg-v0')
    #env = DummyVecEnv([lambda: env])
    env = SubprocVecEnv([lambda: gym.make('tensegrity:TensLeg-v0') for i in range(n_cpu)])
    
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./results')
    if train:
        model.learn(total_timesteps=750000)
        model.save("tppo2_cartpole")
    else:
        model.load("tppo2_cartpole")
    env2 = gym.make('tensegrity:TensLeg-v0')
    obs=env2.reset()
    i = 0
    lin_vel  =[]
    while i in range(0,5000):
        #action, _states = model.predict(obs) 
        action = env2.action_space.sample()
        obs, rewards, dones, info = env2.step(action)
        print(action)
        lin_vel.append(info['reward_linvel'])
        print(info)
        i = i+1
        env2.render(mode="human")
    plt.plot(lin_vel)
    plt.show()
if __name__ == '__main__':
	main()


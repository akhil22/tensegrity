import gym 
import mujoco_py
from gym.envs.mujoco import mujoco_env as mj
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
#from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, TRPO
from stable_baselines.bench import Monitor
import os
import numpy as np
import matplotlib.pyplot as plt
def main():
    train = True
    cont_train = True
    n_cpu = 16
    env = SubprocVecEnv([lambda: gym.make('tensegrity:TensLeg-v1') for i in range(n_cpu)])
    train_num = 2
    stringp =  f"tppo2_tens_sep_legs_run{train_num}"
    stringp_sav = f"tppo2_tens_sep_legs_run{train_num+1}"
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./results', cliprange=0.2, learning_rate=0.00025, ent_coef=0.000001)
    if train:
        if cont_train:
            model.load_parameters(load_path_or_dict=stringp, exact_match=True)
        model.learn(total_timesteps=25250000)
        print('saving model')
        model.save(stringp_sav)
    else:
        print('loading model')
        model = PPO2.load(stringp)
    env2 = gym.make('tensegrity:TensLeg-v1')
    obs=env2.reset()
    i = 0
    lin_vel  =[]
    while i in range(0,50000):
        action, _states = model.predict(obs) 
        #action = env2.action_space.sample()
        obs, rewards, dones, info = env2.step(action)
        if dones:
            obs = env2.reset()
            continue
        lin_vel.append(rewards)
        print(f"rew = {rewards}, lin_vel = {info['reward_linvel']}, height = {info['height_rew']}, joint_c = {info['reward_impact']}, xorc = {info['xorc']}, sensr = {info['sd']} lfoot = {info['lfoot']}, rfoot = {info['rfoot']}, low_knee = {info['lknee']} ctrl={info['ctrl']} DOF = {info['DOF']}")
        i = i+1
        env2.render(mode="human")
    periods = 1000
    weights = np.ones(periods)/periods
    lin_vel_avg = np.convolve(lin_vel, weights, mode='valid')
    plt.plot(lin_vel_avg)
    plt.show()
if __name__ == '__main__':
	main()


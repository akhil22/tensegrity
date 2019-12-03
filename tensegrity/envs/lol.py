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
import matplotlib.pyplot as plt
def main():
    train = True
    cont_train = False
    n_cpu = 8
    env = SubprocVecEnv([lambda: gym.make('tensegrity:TensLeg-v0') for i in range(n_cpu)])
    #env= gym.make('tensegrity:TensLeg-v0')
    #env = DummyVecEnv([lambda: env])
   # env = SubprocVecEnv([lambda: gym.make('HalfCheetah-v2') for i in range(n_cpu)])
   # env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
    
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./results', cliprange=0.2, learning_rate=0.000025, ent_coef=0.01)
    if train:
        if cont_train:
            model.load_parameters(load_path_or_dict="tppo2_cartpole_fall", exact_match=True)
        model.learn(total_timesteps=1750000)
        print('saving model')
        model.save("tppo2_cartpole_fall2")
       # del model
    else:
        print('loading model')
        model = PPO2.load("tppo2_cartpole_fall2")
    #modeload("tppo2_cartpole")
    env2 = gym.make('tensegrity:TensLeg-v0')
   # env2 = gym.make('HalfCheetah-v2')
   # env2 = gym.make('CartPole-v1')
    obs=env2.reset()
    i = 0
    lin_vel  =[]
    while i in range(0,5000):
        action, _states = model.predict(obs) 
        #action = env2.action_space.sample()
        obs, rewards, dones, info = env2.step(action)
      #  print(action)
        lin_vel.append(rewards)
      #  print('sdata')
        print(info['reward_linvel'], info['height_rew'], info['reward_impact'], info['sd'])
        i = i+1
        env2.render(mode="human")
    plt.plot(lin_vel)
    plt.show()
if __name__ == '__main__':
	main()


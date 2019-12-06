# Tensegrity Leg Control Using Reinforcement learning 
This Project is developed as the final course project for the course ECEN 689 603, Reinforcement learning, Texas A&M University.

The project includes Mujoco Model of a tensegrity leg and its corresponding gym environments using openai gym
along with these we provide training files to train the tensegrity leg model to walk using 
[stable baselines](https://github.com/hill-a/stable-baselines) PPPO2 algorithm.

Chekcout the [youtube video](https://www.youtube.com/watch?v=rYjT0UK_1aM) of Tensegrity Leg walking using PPO2. 

## Installation
 ### Dependencies
  - Python 3.6
  - [mujoco-py](https://github.com/openai/mujoco-py).
  - [stable baselines](https://github.com/hill-a/stable-baselines).
 ### Install tensegrity
  After installing the dependencies install tensegrity- 
  ```python
  pip install -e tensegrity
  ```
## Training
  ```python
   python tensegrity/env/sep_leg_train.py
  ```

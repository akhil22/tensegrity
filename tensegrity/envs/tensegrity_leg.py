import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
cwd = os.path.dirname(os.path.abspath(__file__))
class TensLeg(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, cwd+'/tensegrity_legv3.xml', 5)
        utils.EzPickle.__init__(self)
        self.reward_range = (-float('inf'), float('inf')) 

    def _mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]
#         ob = self._get_obs()
#         reward_ctrl = - 0.1 * np.square(action).sum()
#         reward_run = (xposafter - xposbefore)/self.dt
#         reward = reward_ctrl + reward_run
#         done = False
#         return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def step(self, action):
        pos_before = self._mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = self._mass_center(self.model, self.sim)
#         alive_bonus = 5.0
        data = self.sim.data
#        height = self.named.xipos['torso1', 'x']
#        foot   = self.named.xipos['foot', 'x']
#        hop_cost = 400*(height-foot)
        lin_vel_x = self.sim.data.get_body_xvelp("torso1")
        hip_height_cost = 0*self.sim.data.get_body_xpos("Lower_leg")[0]
        lin_vel_x_low = self.sim.data.get_body_xvelp("Lower_leg")[1]
        hip_height_cost2 = 30*self.sim.data.get_body_xpos("body_bar00L")[0]
        hip_height_cost3 = self.sim.data.get_body_xpos("torso1")[0]
        #hip_height_cost = hip_height_cost*hip_height_cost + hip_height_cost2*hip_height_cost2
        hip_height_cost = hip_height_cost + hip_height_cost2 + hip_height_cost3
        jcost = self.sim.data.get_joint_qvel("t2bar00L")*2
       # hip_height_cost = np.square(2-data.xipos[1, 0])
#         hip_height_cost = 0
       # lin_vel_cost = 20 * (pos_after[2] - pos_before[2]) / self.dt
        lin_vel_cost = (5*lin_vel_x[1] + 0*lin_vel_x[0]*lin_vel_x[0] -  5*lin_vel_x_low)
        quad_ctrl_cost = 0.01 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-9 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = 0 * lin_vel_cost - quad_ctrl_cost - quad_impact_cost - 10 * hip_height_cost + jcost*jcost*0
        reward = lin_vel_cost*hip_height_cost
     #   reward = 5 * lin_vel_cost - quad_ctrl_cost - quad_impact_cost
        qpos = self.sim.data.qpos
        ob = self._get_obs()
        done = False
        DOF = self.model.body_dofnum
#         done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return ob, reward, done, dict(reward_linvel=hip_height_cost, reward_quadctrl=-quad_ctrl_cost, 
                                      reward_impact=-quad_impact_cost, joint_cost = jcost*jcost, DOF = DOF)

#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[1:],
#             self.sim.data.qvel.flat,
#         ])

#     def _get_obs(self): 
#         data = self.sim.data
#         return np.concatenate([self.model.body_dofnum.flat, data.qpos.flat, 
#                                data.qvel.flat, data.cinert.flat, data.cvel.flat, data.qfrc_actuator.flat, data.cfrc_ext.flat])
    def _get_obs(self): 
        data = self.sim.data
        return np.concatenate([data.qpos.flat, data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
       # self.viewer.cam.trackbodyid = 1
         self.viewer.cam.distance = self.model.stat.extent * 0.75
      #  self.viewer.cam.lookat[2] = 0
      #  self.viewer.cam.elevation = 0'''

   # def render(self, mode='human', close=False):
    #    super().render()
    	

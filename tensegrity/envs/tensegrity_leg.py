import numpy as np
import math
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
cwd = os.path.dirname(os.path.abspath(__file__))
class Sdata:
    def __init__(self):
        self.n_frames = 0
        self.prev_pos = 0
class TensLeg(mujoco_env.MujocoEnv, utils.EzPickle, Sdata):
    def __init__(self):
        Sdata.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, cwd+'/tensegrity_legv7.xml', 5)
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
        vel = (pos_after - pos_before)/self.dt
#         alive_bonus = 5.0
        data = self.sim.data
#        height = self.named.xipos['torso1', 'x']
#        foot   = self.named.xipos['foot', 'x']
#        hop_cost = 400*(height-foot)
        lin_vel_x = self.sim.data.get_body_xvelp("torso1")
        lin_vel_x = self.sim.data.get_geom_xvelp("torso")
        hip_height_cost = 0*self.sim.data.get_body_xpos("Lower_leg_left")[0]
       # hip_height_cost = 0*self.sim.data.get_body_xpos("Lower_leg_left")[0]
        #lin_vel_x_low = self.sim.data.get_body_xvelp("Lower_leg_left")[1]
        lin_vel_x_low = self.sim.data.get_body_xvelp("Lower_leg_left")[1] + self.sim.data.get_body_xvelp("Lower_leg_right")[1]
        hip_height_cost2 = self.sim.data.get_body_xpos("body_bar00L")[0] + self.sim.data.get_body_xpos("body_bar00R")[0]
        hip_height_cost3 = self.sim.data.get_body_xpos("body_bar00L")[0] + self.sim.data.get_body_xpos("body_bar00R")[0]
       # hip_height_cost3 = self.sim.data.get_body_xpos("torso1")[0]
        #hip_height_cost = hip_height_cost*hip_height_cost + hip_height_cost2*hip_height_cost2
        hip_height_cost = hip_height_cost3 + hip_height_cost2 + hip_height_cost
        hip_height_cost = self.sim.data.get_geom_xpos("torso")[0]
        torso_angle = self.sim.data.get_geom_xmat("torso")[0][0]
        #print("fdajfkldsaj", torso_angle)
        jcost1 = self.sim.data.get_joint_qvel("t2bar00L")
        jcost2 = self.sim.data.get_joint_qvel("t2bar00R")
       # hip_height_cost = np.square(2-data.xipos[1, 0])
#         hip_height_cost = 0
       # lin_vel_cost = 20 * (pos_after[2] - pos_before[2]) / self.dt
       # sensor_data = self.sim.data.sensordata
        lin_vel_cost = (0*vel + lin_vel_x[1] + 0*lin_vel_x[0]*lin_vel_x[0] +  0*lin_vel_x_low)
        quad_ctrl_cost = 0.01 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-1 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
       # reward = 0 * lin_vel_cost - quad_ctrl_cost - quad_impact_cost - 10 * hip_height_cost + jcost*jcost*0
        #xr_cost = ((sensor_data[0] > 0) and not (sensor_data[6] > 0)) or ( not(sensor_data[0] > 0) and (sensor_data[6] > 0))
        knee_upper_left = self.sim.data.get_site_xpos("b3L")[0]
        knee_upper_right = self.sim.data.get_site_xpos("b3R")[0]
        low_knee = (knee_upper_left < -0.4) or ( knee_upper_right < -0.4 ) 
        #reward = lin_vel_cost*100 + 10*hip_height_cost - (sensor_data[2] > 0)*50 - (sensor_data[4] > 0)*50 + xr_cost*0 - quad_impact_cost - (sensor_data[8] > 0)*50 - (sensor_data[10] > 0)*50  + 0*(jcost1*jcost2 < 0)
       # lfoot_vel = self.sim.data.get_geom_xvelp("bar2Lfoot")[1]
       # rfoot_vel = self.sim.data.get_geom_xvelp("bar1Rfoot")[1] 
        xr_rew = (jcost1 >=0) and (jcost1 <= 0.01) and (jcost2*jcost2 > 0)
        xr_rew = xr_rew and (jcost2 >=0) and (jcost2 <= 0.01) and (jcost1*jcost1 > 0)
        xr_cost = xr_rew*(jcost1**2 + jcost2**2)
#        xr_rew = xr_rew or ((rfoot_vel >= -0.3) and (rfoot_vel <= -0.25) and (lfoot_vel > -0.2))
        h_cost = 0
     #   reward = 5 * lin_vel_cost - quad_ctrl_cost - quad_impact_cost
        #reward = lin_vel_cost*100
        qpos = self.sim.data.qpos
        ob = self._get_obs()
       # done = (sensor_data[2] > 0 ) or (sensor_data[4] > 0) or (sensor_data[8] > 0) or (sensor_data[10] > 0)
        done = False
        theta = math.acos(torso_angle)
        h_cost = -np.exp(theta*180/math.pi)
        '''if hip_height_cost < -0.2:
            self.n_frames+=1
            done = True
            return ob, -100, done, dict(reward_linvel=vel*100, height_rew=10*hip_height_cost,reward_impact= -1000*((jcost1*jcost2) > 0), joint_cost = 1000*jcost1*jcost2, DOF = 8989, sd = sensor_data, xorc = xr_cost, lfoot = lfoot_vel, rfoot = rfoot_vel, lknee = -1000*low_knee, ctrl=quad_ctrl_cost)


        if hip_height_cost < 0.52:
            h_cost = -np.exp(10/(hip_height_cost+0.25)) 
        else:
            h_cost = np.exp((-0.4 + hip_height_cost)*20)'''
         
	#reward = lin_vel_cost*100 + 10* hip_height_cost + 1000 * jcost1 * jcost1 - 0*(low_knee) + 0*quad_ctrl_cost
        cur_pos = self.sim.data.get_geom_xpos("torso")[1]
        #reward = lin_vel_cost*100 + 10* hip_height_cost + 0 * jcost1 * jcost1 - 0*(low_knee) + 0*quad_ctrl_cost + np.exp(2*(cur_pos - self.prev_pos))
        '''if lin_vel_cost > 1.0:
            reward = 3
        else:
            reward = 1  '''
        reward = lin_vel_cost
        DOF = self.model.body_dofnum
        if not (self.n_frames % 5000) and self.n_frames != 0:
            done = True
            cur_pos = self.sim.data.get_geom_xpos("torso")[1]
            reward = cur_pos-self.prev_pos
            if cur_pos - self.prev_pos < 0.2:
                done = True 
            #self.prev_pos=cur_pos
            self.prev_pos=0
        self.n_frames+=1
        return ob, reward, done, dict(reward_linvel=lin_vel_cost*100, height_rew=10*hip_height_cost, 
                                      reward_impact= -1000*((jcost1*jcost2) > 0), joint_cost = 1000*jcost1*jcost2, DOF =torso_angle, sd = 100, xorc = xr_cost, lfoot = 100, rfoot = 100, lknee = -1000*low_knee, ctrl=quad_ctrl_cost)

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
    	

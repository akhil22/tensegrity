import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
cwd = os.path.dirname(os.path.abspath(__file__))
class TensLeg(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, cwd+'/tensegrity_legv2.xml', 5)
        utils.EzPickle.__init__(self)
        self.reward_range = (-float('inf'), float('inf')) 
        metadata = {'render.modes': ['human']}
        
        """Soft indicator function evaluating whether a number is within bounds."""
        # The value returned by tolerance() at `margin` distance from `bounds` interval.
        self._DEFAULT_VALUE_AT_MARGIN = 0.1

        self._CONTROL_TIMESTEP = .02  # (Seconds)

        # Default duration of an episode, in seconds.
        self._DEFAULT_TIME_LIMIT = 20

        # Minimal height of torso over foot above which stand reward is 1.
        self._STAND_HEIGHT = 0.6

        # Hopping speed above which hop reward is 1.
        self._HOP_SPEED = 2

    def _mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]
    

    def _sigmoids(self, x, value_at_1, sigmoid):
        """
        Returns 1 when `x` == 0, between 0 and 1 otherwise.
        Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.
        Returns:
        A numpy array with values between 0.0 and 1.0.
        Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
          `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
        """
        if sigmoid in ('cosine', 'linear', 'quadratic'):
            if not 0 <= value_at_1 < 1:
                raise ValueError('`value_at_1` must be nonnegative and smaller than 1, got {}.'.format(value_at_1))
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError('`value_at_1` must be strictly between 0 and 1, got {}.'.format(value_at_1))

        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x*scale)**2)

        elif sigmoid == 'hyperbolic':
            scale = np.arccosh(1/value_at_1)
            return 1 / np.cosh(x*scale)

        elif sigmoid == 'long_tail':
            scale = np.sqrt(1/value_at_1 - 1)
            return 1 / ((x*scale)**2 + 1)

        elif sigmoid == 'cosine':
            scale = np.arccos(2*value_at_1 - 1) / np.pi
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi*scaled_x))/2, 0.0)

        elif sigmoid == 'linear':
            scale = 1-value_at_1
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

        elif sigmoid == 'quadratic':
            scale = np.sqrt(1-value_at_1)
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

        elif sigmoid == 'tanh_squared':
            scale = np.arctanh(np.sqrt(1-value_at_1))
            return 1 - np.tanh(x*scale)**2

        else:
            raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


    def _tolerance(self, x, value_at_margin, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian'):
        """
        Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
        Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
          the target interval. These can be infinite if the interval is unbounded
          at one or both ends, or they can be equal to one another if the target
          value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
          `x` moves out-of-bounds.
          * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
          * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
           'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
          the distance from `x` to the nearest bound is equal to `margin`. Ignored
          if `margin == 0`.
        Returns:
        A float or numpy array with values between 0.0 and 1.0.
        Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
        """
        value_at_margin=0.1
        lower, upper = bounds
        if lower > upper:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin < 0:
            raise ValueError('`margin` must be non-negative.')
        in_bounds = np.logical_and(lower <= x, x <= upper)
        if margin == 0:
            value = np.where(in_bounds, 1.0, 0.0)
        else:
            d = np.where(x < lower, lower - x, x - upper) / margin
            value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))

        return float(value) if np.isscalar(x) else value

    def step(self, action):
        check = True
        if check:
            pos_before = self._mass_center(self.model, self.sim)
            self.do_simulation(action, self.frame_skip)
            pos_after = self._mass_center(self.model, self.sim)
            data = self.sim.data
            lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
            quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            quad_impact_cost = min(quad_impact_cost, 10)
            hip_height_ground = data.xipos[1, 0]
            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost
            ob = self._get_obs()
            done = False
            return ob, reward, done, dict(reward_linvel = lin_vel_cost, reward_quadctrl = -quad_ctrl_cost, 
                                          reward_impact = -quad_impact_cost)
        else:
            data = self.sim.data
            pos_before = self._mass_center(self.model, self.sim)
            self.do_simulation(action, self.frame_skip)
            pos_after = self._mass_center(self.model, self.sim)
            lin_vel = (pos_after - pos_before) / self.dt
            hip_height_ground = data.xipos[1, 0]
            standing = self._tolerance(hip_height_ground,0.1, bounds=(0.6, 2))
            hopping = self._tolerance(lin_vel, 0.1, bounds=(2, float('inf')), 
                                     margin=2/2, sigmoid='linear')
            reward = standing * hopping
            ob = self._get_obs()
            done = False
            return ob, reward, done, dict(reward_standing = standing, reward_hopping = hopping)

    def _get_obs(self): 
        data = self.sim.data
        return np.concatenate([data.qpos.flat, data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def render(self, mode='human', close=False):
        super(TensLeg, self).render(mode=mode)

from gym.envs.registration import register

register(
  id='TensLeg-v0',
  entry_point='tensegrity.model.tensegrity_leg.TensLeg',
)

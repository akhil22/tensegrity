from gym.envs.registration import register
register(
  id='TensLeg-v0',
  entry_point='tensegrity.envs:TensLeg',
)
register(
  id='TensLeg-v1',
  entry_point='tensegrity.envs:TensLegSep',
)

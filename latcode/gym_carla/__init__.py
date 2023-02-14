from gym.envs.registration import register

register(
    id='carla-v1',
    entry_point='gym_carla.envs:CarlaEnv',
)


register(
    id='carla-v2',
    entry_point='gym_carla.envs:CarlaEnv2',
) 

register(
    id='carla-v10',
    entry_point='gym_carla.envs:CarlaEnv10',
) 



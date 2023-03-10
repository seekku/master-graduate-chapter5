#!/usr/bin/env python

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

from collections import deque

import gym
from gym import spaces
from gym.utils import seeding
import carla
import math

#parameter
init_s = 0
end_s = -10   #there should be consistent to env  +/-

Delta = 21


def solve_matrix(l0,l1):
    S = np.array([[init_s**5,init_s**4,init_s**3,init_s**2,init_s,1],
                  [5*init_s**4,4*init_s**3,3*init_s**2,2*init_s,1,0],
                  [20*init_s**3,12*init_s**2,6*init_s,2,0,0],
                  [end_s**5,end_s**4,end_s**3,end_s**2,end_s,1],
                  [5*end_s**4,4*end_s**3,3*end_s**2,2*end_s,1,0],
                  [20*end_s**3,12*end_s**2,6*end_s,2,0,0]])
    l = np.array([[l0],[0],[0],[l1],[0],[0]])  #l的偏移
    a = np.linalg.solve(S,l)        #权重系数

    return a


def calculate_point(a):
    s_list = []
    l_list = []
    for i in range(Delta):
        s = init_s + i*(end_s-init_s)/((Delta-1)*1.00)  
        l = a[0]*s**5+a[1]*s**4+a[2]*s**3+a[3]*s**2+a[4]*s+a[5]
        s_list.append(s)
        l_list.append(l)
    return s_list,l_list


def XYtoSL(x0,y0):
    pass


def SLtoXY(x0,y0,s,l):
    x = []
    y = []
    for i in range(len(s)):
        x.append(s[i]+x0)
        y.append(l[i]+y0)

    return x,y



class CarlaEnv11(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters

    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.desired_speed = params['desired_speed']
    self.dests = [[145,58.910496,0.275307]]

    self.visualize = True  #是否可视化轨迹点
    self.kp = 0.75
    self.ki = 0.4
    self.kd = 0.02

    self.collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')


    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(10.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)
    self.spectator = self.world.get_spectator()
    

    # # Get spawn points
    # self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    # self.walker_spawn_points = []


    # Create the ego vehicle blueprint
    self.ego_bp = random.choice(self.world.get_blueprint_library().filter("vehicle.lincoln*"))
    self.ego_bp.set_attribute('color', "255,0,0")
    self.surround_bp =random.choice(self.world.get_blueprint_library().filter("vehicle.carlamotors*"))
    self.surround_bp.set_attribute('color',"255,128,0")

    # # Camera sensor
    self.camera_img = np.zeros((384, 216, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(384))
    self.camera_bp.set_attribute('image_size_y', str(216))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # self.camera_img2 = np.zeros((1920, 1080, 3), dtype=np.uint8)
    # self.camera_trans2 = carla.Transform(carla.Location(x=0.8, z=1.7))
    # self.camera_bp2 = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # # Modify the attributes of the blueprint to set image resolution and field of view.
    # self.camera_bp2.set_attribute('image_size_x', str(1920))
    # self.camera_bp2.set_attribute('image_size_y', str(1080))
    # self.camera_bp2.set_attribute('fov', '110')
    # # Set the time in seconds between sensor captures
    # self.camera_bp2.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # # Initialize the renderer
    # self._init_renderer()

    # # Get pixel grid points
    # if self.pixor:
    #   x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
    #   x, y = x.flatten(), y.flatten()
    #   self.pixel_grid = np.vstack((x, y)).T

  def reset(self):
    # Clear sensor objects
    self.collision = False

    transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.collision_sensor_detection = self.world.spawn_actor(self.collision_sensor, transform, attach_to=self.ego)
    def callback_ego(event):
            self.collision = True
            print('collision!!!!!!!')
    self.collision_sensor_detection.listen(callback_ego)
    self.lidar_sensor = None
    self.camera_sensor = None
    # self.camera_sensor2 = None
    self.path = []  #规划的路径点存储(可能会有重复，但应该没关系)。
    self._lon_ebuffer = deque(maxlen=10)

    self.location_flag = None
    self.info = True
    self.lat_count = 1  #用于计算几次横向决策，初始就要先来一次。
    self.current_L = 0  #起始位置在车道中心
    self.target_L = 0  #目标位置在车道中心

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb','sensor.camera.semantic_segmentation', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)
    spaw_points = self.world.get_map().get_spawn_points()

    self.random_ego_x = 181.5+np.random.uniform(-10,10)
    self.vehicle_spawn_points0 = carla.Transform(carla.Location(x=self.random_ego_x, y=59, z=0.275307), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
    # self.vehicle_spawn_points0 = carla.Transform(
    #   carla.Location(x=181.5, y=59, z=0.275307),
    #   carla.Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000))
    self.vehicle_spawn_points1 = carla.Transform(carla.Location(x=158, y=55, z=0.275307), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
    self.ego = self.world.spawn_actor(self.ego_bp,self.vehicle_spawn_points0)

    self.surround = self.world.try_spawn_actor(self.surround_bp,self.vehicle_spawn_points1)
    self.surround.set_autopilot(False)


    # spawing a walker
    blueprint = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    spawn_point = carla.Transform(carla.Location(x=151.5,y=53,z=0.275307),carla.Rotation(pitch=0.000000, yaw=90.852554, roll=0.000000))
    # spawn_points = self.world.get_map().get_spawn_points()
    # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    self.person = self.world.spawn_actor(blueprint, spawn_point)

    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
    walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), self.person)
    # start walker
    walker_controller_actor.start()
    # set walk to random point
    # walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
    walker_controller_actor.go_to_location(carla.Location(x=151.5, y=68, z=0.275307))
    # random max speed
    walker_controller_actor.set_max_speed(1.5)  # max speed between 1 and 2 (default is 1.4 m/s)

    # self.ego.set_target_velocity(carla.Vector3D(-self.desired_speed,0,0))
    self.ego.set_target_velocity(carla.Vector3D(-self.desired_speed,0,0))
    # # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: get_camera_img(data))
    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
      image = np.reshape(array, (data.height, data.width, 4))

      # Get the r channel
      sem = image[:, :, 2]
      # print(sem)
      m = len(sem[0,:])
      if self.location_flag == None:
        for i in range(len(sem[:,0])):
          for j in range(int(m/2)):
            if sem[i][j+int(m/2)] == 4:
              self.location_flag = True

      # print(self.location_flag)

    # Update timesteps
    self.time_step=0
    self.reset_step+=1
    # self.spectator.set_transform(carla.Transform(carla.Location(x=self.hero_location.x, y=self.hero_location.y, z = 40)))  # 89.9 to avoid gimbal lock

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # Set ego information for render

    return self._get_obs()
  
  def step(self, act):
    # Calculate acceleration and steering
    action = act[0]
    lat_action = act[1]


    throttle = 0
    brake = 0

    if action == 0:
      brake = 0.5
    elif action == 1:
      brake = 0.25
    elif action == 2:
      throttle = 0.2
    elif action == 3:
      throttle = 0.4
    elif action == 4:
      throttle = 0.6
    else:
      throttle = 0.8
    
    ego_location = carla.Location()
    ego_location.x = self.ego.get_transform().location.x
    ego_location.y = self.ego.get_transform().location.y
    ego_location.z = self.ego.get_transform().location.z
    ego_yaw = self.ego.get_transform().rotation.yaw
    ego_velcocity = math.sqrt(self.ego.get_velocity().x**2+self.ego.get_velocity().y**2)
    
    
    
    if self.info:  #这个label是不是可以换个和别的统一起来？
      self.current_L = self.target_L

      if lat_action == 0:
        self.target_L = -1
      elif lat_action == 1:
        self.target_L = -0.5
      elif lat_action == 2:
        self.target_L = 0
      elif lat_action == 3:
        self.target_L = 0.5
      else:
        self.target_L = 1

      # to visualize

      # carla.Location(x=181.5, y=59, z=0.275307)  #车辆的起初位置。
      # planner.XYtoSL(x0=181.5,y0=59)
      SL_matrix = solve_matrix(self.current_L,self.target_L)
      S,L = calculate_point(SL_matrix)
      self.X,self.Y = SLtoXY(x0=self.random_ego_x+(self.lat_count-1)*end_s,y0=59,s=S,l=L)
      
      if self.visualize:
        for i in range(len(self.X)):
          self.path.append([self.X[i],self.Y[i]])
          debug_point = carla.Location()
          # print(type(debug_point.x))
          debug_point.x = float(self.X[i])
          debug_point.y = float(self.Y[i])
          debug_point.z = self.ego.get_transform().location.z+0.1
          ###planning point
          self.world.debug.draw_point(debug_point,0.05,carla.Color(255,0,0),2)
    if self.info == True:
      self.info = False
    
    #横向控制PID
    target_point = [self.ego.get_transform().location.x,self.ego.get_transform().location.y]
    for index in range(len(self.path)):
      if self.path[index][0]>self.ego.get_transform().location.x:
        continue
      else:
        target_point = [self.path[index][0],self.path[index][1]]
    v_begin = self.ego.get_transform().location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(self.ego.get_transform().rotation.yaw)),y=math.sin(math.radians(self.ego.get_transform().rotation.yaw)))
    v_vec = np.array([v_end.x-v_begin.x,v_end.y-v_begin.y,0.0])
    w_vec = np.array([target_point[0]-v_begin.x, target_point[1]-v_begin.y , 0.0])
    # print(v_vec.shape)
    # print(v_begin.shape)
    _dot = math.acos(np.clip(np.dot(w_vec,v_vec)/(np.linalg.norm(w_vec)*np.linalg.norm(v_vec)),-1.0,1.0))
    # if  w_vec!=np.array([0,0,0]):
    w_vec[1] = float(w_vec[1])
    w_vec = np.array([w_vec[0],w_vec[1],w_vec[2]])
    #   w_vec = np.array([w_vec[0],w_vec[1][0],w_vec[2]])  #这里是为了修复bug
    _cross = np.cross(v_vec,w_vec)

    if _cross[2]<0:
      _dot *= -1
    self._lon_ebuffer.append(_dot)
    if(len(self._lon_ebuffer))>=2:
      _de = (self._lon_ebuffer[-1] - self._lon_ebuffer[-2])/self.dt
      _ie = sum(self._lon_ebuffer) * self.dt
    else:
      _de = 0.0
      _ie = 0.0
    
    steer = float(np.clip(self.kp*_dot+self.kd*_de+self.ki*_ie,-1.0,1.0))

    
    # Apply control
    act = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    self.ego.apply_control(act)
    self.world.tick()


    # Update timest
    self.spectator.set_transform(carla.Transform(carla.Location(x=ego_location.x - 10, y=ego_location.y, z = 60),
                                carla.Rotation(yaw = 90, pitch = -90, roll = 0)))
    self.time_step += 1
    self.total_step += 1

    # if self.info == True:
    #   self.info = False  #这个信息用于传递信号，应该如何传递信号呢？
    if ego_location.x - self.random_ego_x < end_s * self.lat_count + 2:  #这里需要记住一下，提前0.5m规划一下。
      self.info = True
      self.lat_count += 1


    return (self._get_obs(), self._get_reward(), self._terminal(),[self.info,self._lat_get_reward()])  

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # print(self.X[i],self.Y[i])
  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)


  def _get_obs(self):
    """Get the observations."""


    #State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y

    surround_x = self.surround.get_transform().location.x
    surround_y = self.surround.get_transform().location.y

    person_x = 150
    person_y = 50
    if self.location_flag:
      person_x = self.person.get_transform().location.x
      person_y = self.person.get_transform().location.y
    person_v = self.person.get_velocity()
    egovehicle_v = self.ego.get_velocity()
    # print(egovehicle_v.x)
    obs = [surround_x-ego_x,surround_y-ego_y,person_x-ego_x,person_y-ego_y,person_v.y,egovehicle_v.x] # relative location

    return obs

  def _get_reward(self):
    """Calculate the step reward."""
    # # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = speed
    if speed > self.desired_speed:
      r_speed = speed - (speed - self.desired_speed)**2
    #
    # reward for collision
    r_collision = 0
    ego_x = self.ego.get_transform().location.x
    ego_y = self.ego.get_transform().location.y

    person_x = self.person.get_transform().location.x
    person_y = self.person.get_transform().location.y


    ### for train
    # if abs(person_x-ego_x)<3 and abs(person_y-ego_y)<2.5:
    #   r_collision = -1
    ### for test
    if self.collision:
      r_collision = -1
    
    r_time = 0
    if self.time_step>self.max_time_episode:
      r_time = -1

    # cost for cceleration
    a = self.ego.get_acceleration()
    r_acc_x = -abs(a.x)
    acc = np.sqrt(a.x**2+a.y**2)
    r_acc = -abs(acc**2)

    ego_x = self.ego.get_transform().location.x
    ego_y = self.ego.get_transform().location.y

    r_success = 0
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<2:
          r_success = 1

    r = 1000 * r_collision + r_speed  + r_acc_x +  500 * r_success + 200 * r_time
    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # # Get ego state
    # ego_x, ego_y = get_pos(self.ego)
    #
    # # If collides
    # if len(self.collision_hist)>0:
    #   return True
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)

    ego_x = self.ego.get_transform().location.x
    ego_y = self.ego.get_transform().location.y

    person_x = self.person.get_transform().location.x
    person_y = self.person.get_transform().location.y


    ##collision  for train
    # if abs(person_x-ego_x)<3 and abs(person_y-ego_y)<2.5:
    #   print(abs(person_x-ego_x),abs(person_y-ego_y))
    #   print("ego vehicle speed:",speed)
    #   return True
    ## coliision for test
    if self.collision:
      print(abs(person_x-ego_x),abs(person_y-ego_y))
      print("ego vehicle speed:",speed)
      return True





    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True



    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<2:
          return True


    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()

  def _lat_get_reward(self):
    #这里就是看换道路
    if  self.current_L == self.target_L:
      r = 1
    elif abs( self.current_L-self.target_L)>1:
      r = -2
    else:
      r = 0

    if self.target_L == 0:
      r += 1

    return r 

  def get_heading_error(self,target_point,target_next_point,current_yaw):
    ego_x = self.ego.get_transform().location.x
    ego_y = self.ego.get_transform().location.x
    target_point_delta_x = target_next_point[0] - ego_x if  target_next_point[0] - ego_x != 0 else 0.000001
    target_point_delta_y = target_next_point[1] - ego_y if  target_next_point[1] - ego_y != 0 else 0.000001
    target_heading  = np.arctan(target_point_delta_y/target_point_delta_x)
    heading_error_mod = divmod((target_heading-current_yaw),np.pi)[1]
    if np.pi/2<heading_error_mod<np.pi:
      heading_error_mod -= np.pi
    return heading_error_mod

  def get_steering_direction(self,v1,v2):
    cross_prod = v1[0]*v2[1]-v1[1]*v2[0]
    if cross_prod >= 0:
      return -1
    return 1
  
  def get_cte_heading_error(self,v):
    return 0

  def calculate_run_distance(self):
    distance = self.random_ego_x - 145 - 1.4
    return distance
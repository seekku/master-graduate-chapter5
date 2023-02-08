import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
import math
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from collections import deque, namedtuple
import time
import gym
import gym_carla
import carla


def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01,collision = 0,success =0,writer = None):
    """

    Params
    ======

    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    output_history = []
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    i_episode = 1
    state = env.reset()
    score = 0
    for frame in range(1, frames + 1):

        action = 0
        next_state, reward, done, _ = env.step(action)
        time.sleep(10)
        

        if done:
            state = env.reset()


    return output_history


# writer = SummaryWriter("runs/" + "FQF99")
seed = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using ", device)

# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# env = gym.make("Intersection-v0")
# env.seed(seed)
# eval_env = gym.make("Intersection-v0")
# eval_env.seed(seed + 1)
# action_size = env.action_space.n
# state_size = env.observation_space.shape

params = {
    'dt': 0.1,  # time interval between two frames
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 1000,  # maximum timesteps per episode
    'desired_speed': 0,  # desired speed (m/s)
}

np.random.seed(seed)
env = gym.make('carla-v10', params=params)

env.seed(seed)
# action_size = env.action_space.n
# state_size = env.observation_space.shape
action_size = 6
state_size = [6]

# set epsilon frames to 0 so no epsilon exploration
eps_fixed = False

t0 = time.time()
final_average100 = run(frames = 10000, eps_fixed=eps_fixed, eps_frames=5000, min_eps=0.05, collision = 0)
t1 = time.time()

print("Training time: {}min".format(round((t1-t0)/60,2)))

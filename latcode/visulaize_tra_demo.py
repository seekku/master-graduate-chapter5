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

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)


class QVN(nn.Module):
    """Quantile Value Network"""

    def __init__(self, state_size, action_size, layer_size, n_step, device, seed, layer_type="ff"):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.K = 32
        self.N = 32
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(device)  # Starting from 0 as in the paper
        self.device = device
        self.head = nn.Linear(self.input_shape[0], layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head, self.ff_1])

    def calc_cos(self, taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1) * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos

    def forward(self, input):
        """Calculate the state embeddings"""
        return torch.relu(self.head(input))

    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]

        """
        if embedding == None:
            x = self.forward(input)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size)


class FPN(nn.Module):
    """Fraction proposal network"""

    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])

    def forward(self, x):
        """
        Calculates tau, tau_ and the entropy

        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """
        q = self.softmax(self.ff(x))
        q_probs = q.exp()
        # print("q_probs:",q_probs)
        taus = torch.cumsum(q_probs, dim=1)
        # print("tau:",taus)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(device), taus), dim=1)
        # print("tau2:",taus)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        # print("taus_:",taus_)

        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        # print("entropy:",entropy)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)

        return taus, taus_, entropy


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma ** idx * self.n_step_buffer[idx][2]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], \
               self.n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step
        self.entropy_coeff = 0.001
        self.N = 32
        self.action_step = 4
        self.last_action = None

        # FQF-Network
        self.qnetwork_local = QVN(state_size, action_size, layer_size, n_step, device, seed).to(device)
        self.qnetwork_target = QVN(state_size, action_size, layer_size, n_step, device, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000,gamma=0.2,last_epoch=-1)
        print(self.qnetwork_local)

        self.FPN = FPN(layer_size, seed, 32, device).to(device)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=LR * 0.000001, alpha=0.95, eps=0.00001)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            loss = self.learn(experiences)
            self.Q_updates += 1
            # writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy"""
        # Epsilon-greedy action selection
        if random.random() > eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                embedding = self.qnetwork_local.forward(state)
                taus, taus_, entropy = self.FPN(embedding)
                F_Z = self.qnetwork_local.get_quantiles(state, taus_, embedding)
                action_values = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z).sum(1)
                assert action_values.shape == (1, self.action_size)

            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())

        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))

        assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

        # calc fractional loss
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N - 1, 1))

        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean()
        frac_loss += entropy_loss

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            next_state_embedding_loc = self.qnetwork_local.forward(next_states)
            n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
            F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)
            Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1)) * F_Z_next).sum(1)
            action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)

            next_state_embedding = self.qnetwork_target.forward(next_states)
            F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
            Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(
                1, 2)

            Q_targets = rewards.unsqueeze(-1) + (
                        self.GAMMA ** self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)

        # print("taus:",taus.shape)
        # print("td_error",td_error.shape)
        # print("huber_l",huber_l.shape)
        quantil_l = abs(taus_.unsqueeze(-1) - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)
        loss = loss.mean()

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()

        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        self.lr_scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


def calc_fraction_loss(FZ_, FZ, taus):
    """calculate the loss for the fraction proposal network """

    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:]
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(
        BATCH_SIZE, 31)
    assert not gradients.requires_grad
    loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss


# def eval_runs(eps, frame):
#     """
#     Makes an evaluation run with the current epsilon
#     """
#     reward_batch = []
#     for i in range(5):
#         state = eval_env.reset()
#         rewards = 0
#         while True:
#             action = agent.act(state, eps)
#             state, reward, done, _ = eval_env.step(action)
#             rewards += reward
#             if done:
#                 break
#         reward_batch.append(rewards)
#
#     # writer.add_scalar("Reward", np.mean(reward_batch), frame)


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

        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame * (1 / eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps * ((frame - eps_frames) / (frames - eps_frames)), 0.001)

        # evaluation runs
        if frame % 100 == 0:
            print('-'*30)
            print(frame)
            print('-'*30)

        if done:
            if reward< -400:
                collision += 1
            if reward > 10:
                success += 1

            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            if i_episode>=20 or score < -500:
                writer.add_scalar("FQF Distributional RL reward",score,frame)

            print("last time reward:",reward)
            print("this epoch reward:",score)
            # writer.add_scalar("Average100", np.mean(scores_window), frame)
            output_history.append(np.mean(scores_window))
            # print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(i_episode, frame,
            #                                                                                 np.mean(scores_window),
            #                                                                                 eps), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}\tCollision Rate: {:.2f}\t Success Rate:{:.2f}'.format(i_episode, frame, np.mean(scores_window),collision/100,success/100))
                writer.add_scalar("Success Rate",success/100,i_episode)
                writer.add_scalar("collision Rate",collision/100,i_episode)
                collision = 0
                success = 0
            i_episode += 1
            state = env.reset()
            score = 0

    return output_history


writer = SummaryWriter("runs/" + "FQF99")
seed = 99
BUFFER_SIZE = 40000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-2
LR = 1e-3
UPDATE_EVERY = 1
n_step = 1
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

agent = DQN_Agent(state_size=state_size,
                  action_size=action_size,
                  layer_size=256,
                  n_step=n_step,
                  BATCH_SIZE=BATCH_SIZE,
                  BUFFER_SIZE=BUFFER_SIZE,
                  LR=LR,
                  TAU=TAU,
                  GAMMA=GAMMA,
                  UPDATE_EVERY=UPDATE_EVERY,
                  device=device,
                  seed=seed)


# set epsilon frames to 0 so no epsilon exploration
eps_fixed = False

t0 = time.time()
final_average100 = run(frames = 100000, eps_fixed=eps_fixed, eps_frames=5000, min_eps=0.05, collision = 0,writer=writer)
t1 = time.time()

print("Training time: {}min".format(round((t1-t0)/60,2)))

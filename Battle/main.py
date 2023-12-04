import magent2
import config.battle
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.utils import to_categorical
import numpy as np
from torch_geometric.data import Data
from pettingzoo.utils import random_demo
'''
Utilities for converting to graph data
'''
def create_edge_index(state, k=4):
    # state: [num_agents, state_dim]
    num_agents = state.shape[0]
    edge_index = []

    for i in range(num_agents):
        distances = np.sum((state - state[i]) ** 2, axis=1)
        neighbors = np.argsort(distances)[1:k + 1]
        for neighbor in neighbors:
            edge_index.append([i, neighbor.item()])

    return torch.tensor(edge_index).t().contiguous()

def Adjacency(state):
    adj = []
    dis = []
    for j in range(20):
        dis.append([state[j][-2], state[j][-1], j])
    for j in range(20):
        f = []
        for r in range(len(dis)):
            f.append([(dis[r][0] - dis[j][0]) ** 2 + (dis[r][1] - dis[j][1]) ** 2, r])
        f.sort(key=lambda x: x[0])
        y = []
        for r in range(4):
            y.append(f[r][1])
        y = to_categorical(y, num_classes=20)
        adj.append(y)
    return adj

def observation(state1, state2):
    state = []
    for j in range(20):
        state.append(np.hstack(
            ((state1[j][0:11, 0:11, 1] - state1[j][0:11, 0:11, 5]).flatten(), state2[j][-1:-3:-1])))
    return state
def process_observations(state1, state2):
    state = []
    for j in range(20):
        state.append(np.hstack(
            ((state1[j][0:11, 0:11, 1] - state1[j][0:11, 0:11, 5]).flatten(), state2[j][-1:-3:-1])))
    state_array = np.array(state)
    state_tensor = torch.tensor(state_array, dtype=torch.float)
    return state_tensor


def create_graph_data(state1, state2):
    x = process_observations(state1, state2)
    edge_index = create_edge_index(state2)
    data = Data(x=x, edge_index=edge_index)
    return data

'''
Network Definitions
'''

class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()
        self.fc1 = nn.Linear(123, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MultiHeadAttn(nn.Module):
    def __init__(self, l=2, d=128, dv=16, dout=128, nv=8):
        super(MultiHeadAttn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d, num_heads=nv)
        self.linear = nn.Linear(d, dout)

    def forward(self, v, k, q):
        attn_output, _ = self.multihead_attn(q, k, v)
        output = self.linear(attn_output)
        return output

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DGN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DGN, self).__init__()
        self.encoder = ObservationEncoder()
        self.attention1 = MultiHeadAttn()
        self.attention2 = MultiHeadAttn()
        self.q_net = QNetwork(input_dim=128 * 3, action_dim=action_dim)

    def forward(self, data):
        # data contains node features and edge indices
        '''
        1) Get feature representation by ObservationEncoder, output
        should be 1x128
        2)
        '''
        feature = self.encoder(data.x)
        attn_output1 = self.attention1(feature, feature, feature)  # q, k, v are all the same
        attn_output2 = self.attention2(attn_output1, attn_output1,
                                       attn_output1)  # q, k, v are all the same
        q_values = self.q_net(feature, attn_output1, attn_output2)
        return q_values

class Agent(object):
    def __init__(self, state_size, action_size, learning_rate, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.model = self.build_model()
        self.target_model = self.build_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def build_model(self):
        model = DGN(state_dim=self.state_size, action_dim=self.action_size)
        return model

    def build_target_model(self):
        model = DGN(state_dim=self.state_size, action_dim=self.action_size)
        return model

    def decide_action(self, data):
        with torch.no_grad():
            action_values = self.model(data)
        return action_values


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.num_experiences = 0
        self.buffer = deque()

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def __len__(self):
        return len(self.buffer)


class Environment(object):
    def __init__(self, agent, replay_buffer, env, batch_size, max_steps, tau, alpha, gamma, N, L):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.env = env
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.N = N
        self.L = L
        self.episode_before_train = 5

    def reset(self):
        return self.env.reset()

    def step(self, handles):
        done = self.env.step()
        n = len(handles)
        next_observation = [[] for _ in range(n)]
        next_observation[0] = self.env.get_observation(handles[0])
        next_data = create_graph_data(next_observation[0][0], next_observation[0][1])
        next_state = next_data
        rewards = self.env.get_reward(handles[0])
        return next_state, rewards, done

    def learn(self, batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in batch:
            state, action, reward, next_state, done = experience
            print(state)
            print(next_state)
            # Convert state and next_state to numpy arrays if they are not already
            state = np.array(state, dtype=np.float32)
            next_state = np.array(next_state, dtype=np.float32)
            states.append(torch.from_numpy(state))
            actions.append(torch.tensor(action, dtype=torch.int64))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            next_states.append(torch.from_numpy(next_state))
            dones.append(torch.tensor(done, dtype=torch.float32))

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        current_q_values = self.agent.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.agent.target_model(next_states).max(1)[0]
        next_q_values[dones] = 0.0
        target_q_values = rewards + self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()

        for target_param, local_param in zip(self.agent.target_model.parameters(),
                                             self.agent.model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    def run_episode(self, episode):
        self.alpha *= 0.996
        if self.alpha < 0.01:
            self.alpha = 0.01
        state = self.reset()
        done = False
        handles = self.env.get_handles()
        self.env.add_agents(handles[0], method="random", n=self.N)
        self.env.add_agents(handles[1], method="random", n=self.L)
        n_actions = self.env.get_action_space(handles[0])[0]
        n = len(handles)
        observations = [[] for _ in range(n)]
        ids = [[] for _ in range(n)]
        action = [[] for _ in range(n)]
        nums = [self.env.get_num(handle) for handle in handles]
        score = 0
        loss = 0
        dead = [0, 0]
        steps = 0
        while not done and steps < self.max_steps:
            steps += 1
            '''
            Obtain observations of agents and use model to decide on next action for handles[0] group
            '''
            observations[0] = self.env.get_observation(handles[0]) # obtain the observations from the N agents
            data = create_graph_data(observations[0][0], observations[0][1])
            state = data
            acts = self.agent.decide_action(data)
            action[0] = np.zeros(self.N,dtype = np.int32)
            for j in range(self.N):
                if np.random.rand() < self.alpha:
                    action[0][j] = random.randrange(n_actions)
                else:
                    action[0][j] = np.argmax(acts[j])
            self.env.set_action(handles[0], action[0])
            '''
            Obtain observations of agents and use pretrained model for their actions
            '''
            observations[1] = self.env.get_observation(handles[1])
            ids[1] = env.get_agent_id(handles[1])
            #acts = tf_model.infer_action (observations[1], ids[1], 'e_greedy')
            acts = action[0]
            self.env.set_action(handles[1], acts)
            '''
            Step to obtain next observation, reward, score
            '''
            next_state, reward, done = self.step(handles)
            score += sum(reward)
            # Add to buffer
            if steps % 3 == 0:
                self.replay_buffer.add(state, action[0], reward, next_state, done)
            # Print dead number, score, loss
            if self.max_steps == steps:
                print(dead[0], end='\t')
                print(dead[1], end='\t')
                print(score / 300, end='\t')
                # f.write(str(dead[i])+'\t'+str(score[i]/300)+'\t')
                # f.write(str(loss/100)+'\n')
                print(loss / 100, end='\n')
            self.env.clear_dead()
            '''
            Add to N Agents (adding agents to the environment randomly)
            '''
            idd = self.N - len(self.env.get_agent_id(handles[0]))
            if idd > 0:
                self.env.add_agents(handles[0], method="random", n=idd)
                dead[0] += idd
            idd = 12 - len(self.env.get_agent_id(handles[1]))
            if idd > 0:
                self.env.add_agents(handles[1], method="random", n=idd)
                dead[1] += idd
            '''
            Training
            '''
            if episode < self.episode_before_train:
                continue
            if steps % 3 != 0:
                continue

            batch = self.replay_buffer.sample(128)
            self.learn(batch)

            # Occasionally render the environment
            if (episode - 1) % 10 == 0:
                self.env.render()

    def train(self, episodes):
        for episode in range(episodes):
            self.run_episode(episode)

if __name__ == "__main__":
    path = "data/battle_model"
    # Environment parameters
    map_size = 11 # 11 x 11 grid
    cfg = config.battle.get_config(map_size)
    env = magent2.GridWorld(cfg, render_mode='human')
    env.set_render_dir("render")
    N = 20 # number of agents
    L = 12 # number of adversaries
    handles = env.get_handles()
    action_size = env.get_action_space(handles[0])[0]
    state_size = 12
    # Hyperparameters
    buffer_size = 10000
    batch_size = 10
    episodes = 2000
    episodes_before_train = 200
    max_steps = 300 # 300 for battle, 120 for jungle
    tau = 0.01
    learning_rate = 10e-4
    alpha = 0.6
    gamma = 0.96 # for jungle/battle gamma = 0.96, for routing gamma = 0.98

    # Create instances
    replay_buffer = ReplayBuffer(buffer_size)
    agent = Agent(state_size, action_size, learning_rate, tau)
    environment = Environment(agent, replay_buffer, env, batch_size, max_steps, tau,
                              alpha, gamma, N, L)

    # Train the agent
    environment.train(episodes)

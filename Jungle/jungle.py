import magent2
import config.eat
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Batch
import logging
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

'''
Utilities for converting to graph data
'''

def create_edge_index(state, k=3):
    # state: [num_agents, state_dim]
    num_agents = state.shape[0]
    edge_index = []

    for i in range(num_agents):
        distances = np.sum((state - state[i]) ** 2, axis=1)
        neighbors = np.argsort(distances)[1:k + 1]
        for neighbor in neighbors:
            edge_index.append([i, neighbor.item()])

    return torch.tensor(edge_index).t().contiguous()

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
        '''
        where to set tau?
        :param l:
        :param d:
        :param dv:
        :param dout:
        :param nv: 8 in paper
        '''
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


class G2ANet(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super(G2ANet, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph attention layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x
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
        model.to(device)
        return model

    def build_target_model(self):
        model = DGN(state_dim=self.state_size, action_dim=self.action_size)
        model.to(device)
        return model

    def decide_action(self, data):
        self.model.to(device)
        self.model.eval()
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
        # Randomly sample batch_size example
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
    def __init__(self, agent, policy, replay_buffer, env, batch_size, max_steps, beta, epsilon, gamma, N, L):
        self.policy = policy
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.env = env
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.N = N
        self.L = L
        self.episode_before_train = 200

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

    def learn(self, batch,step):
        states = [experience[0] for experience in batch]
        actions = [experience[1] for experience in batch]
        rewards = [experience[2] for experience in batch]
        next_states = [experience[3] for experience in batch]
        dones = [experience[4] for experience in batch]
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

        states = Batch.from_data_list(states)
        next_states = Batch.from_data_list(next_states)
        next_states.to(device)

        self.agent.model.eval()

        current_q_values_bell = self.agent.model(states)
        current_q_values_bell = current_q_values_bell.view(self.N, len(batch), action_size)

        self.agent.target_model.to(device)
        self.agent.target_model.eval()

        target_q_values = self.agent.target_model(next_states)
        target_q_values = target_q_values.view(self.N, len(batch), action_size)

        current_q_values = current_q_values_bell

        for k in range(len(batch)):
            if dones[k]:
                for j in range(self.N):
                    # Update Q-value for terminal state
                    current_q_values[j][k][actions[k][j]] = rewards[k][j]
            else:
                for j in range(self.N):
                    # Update Q-value for non-terminal state
                    current_q_values[j][k][actions[k][j]] = rewards[k][j] + self.gamma * torch.max(
                        target_q_values[j][k])

        self.agent.optimizer.zero_grad()
        self.agent.model.train()
        loss = F.mse_loss(current_q_values_bell, current_q_values)
        loss.backward()
        self.agent.optimizer.step()
        #logging.info(f'Training Loss: {loss.item()}')
        target_net_state_dict = self.agent.target_model.state_dict()
        policy_net_state_dict = self.agent.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.beta + target_net_state_dict[
                key] * (1 - self.beta)
        self.agent.target_model.load_state_dict(target_net_state_dict)

    def run_episode(self, episode):
        self.epsilon *= 0.996
        if self.epsilon < 0.01:
            self.epsilon = 0.01
        self.env.reset()
        done = False
        handles = self.env.get_handles()
        self.env.add_agents(handles[0], method="random", n=self.N)
        self.env.add_agents(handles[1], method="random", n=self.L)
        n_actions = self.env.get_action_space(handles[0])[0]
        n = len(handles)
        observations = [[] for _ in range(n)]
        ids = [[] for _ in range(n)]
        action = [[] for _ in range(n)]
        score = 0
        damage = 0
        steps = 0
        episode_rewards = []
        while not done and steps < self.max_steps:
            steps += 1
            '''
            Obtain observations of agents and use model to decide on next action for handles[0] group
            '''
            observations[0] = self.env.get_observation(handles[0]) # obtain the observations from the N agents
            data = create_graph_data(observations[0][0], observations[0][1])
            state = data.to(device)
            acts = self.agent.decide_action(data)
            action[0] = np.zeros(self.N,dtype = np.int32)
            '''
            DenseNet; with chance epsilon, select random action
            '''
            for j in range(self.N):
                if np.random.rand() <= self.epsilon or self.policy == 'random':

                    action[0][j] = random.randrange(n_actions)
                else:
                    action[0][j] = np.argmax(acts[j].cpu().detach().numpy())
            self.env.set_action(handles[0], action[0])
            '''
            Obtain observations of agents and use pretrained model for their actions
            '''
            '''
            Step to obtain next observation, reward, score
            '''
            next_state, reward, done = self.step(handles)
            episode_rewards.append(sum(reward))
            # Add to buffer
            #if steps % 3 == 0:
            self.replay_buffer.add(state, action[0], reward, next_state, done)
            if max_steps == steps:
                damage = 0
                for j_ in range(N):
                    damage = damage + 400 - state.x[j_] * 400
                damage = damage / 20
                #print(score / 300, end='\t')
            damage = 0
            self.env.clear_dead()
            '''
            Training
            '''
            if episode < self.episode_before_train:
                continue
            # Keep adjacency matrix the same for two consecutive timesteps when computing Q-loss
            if steps % 3 != 0:
                continue

            if policy != 'random':
                batch = self.replay_buffer.sample(10)
                self.learn(batch,steps)

            # Occasionally render the environment
            if (episode - 1) % 10 == 0:
                self.env.render()
        total_episode_reward = sum(episode_rewards) / self.max_steps
        mean_episode_reward = total_episode_reward / self.N  # Mean reward per agent
        return mean_episode_reward

    def train(self, episodes):
        mean_reward = []
        for episode in range(episodes):
            logging.info(f'Starting episode {episode + 1}/{episodes}')
            mean_reward.append(self.run_episode(episode))
            logging.info(f'Episode {episode + 1} completed')
            logging.info(
                f'Episode {episode + 1} Summary: Mean Reward: {mean_reward[episode]}')
            # Save the model every 100 episodes or at the end of training
            if (episode + 1) % 100 == 0 or episode + 1 == episodes:
                model_save_path = f'model_episode_{episode + 1}.pth'
                torch.save(self.agent.model.state_dict(), model_save_path)
                plt.figure(figsize=(10, 5))
                plt.plot(mean_reward)
                plt.xlabel('Episodes')
                plt.ylabel('Mean Reward')
                plt.title(f'Mean Reward up to Episode {episode + 1}')
                plt.savefig(f'mean_rewards_{episode + 1}.png')  # Save the plot
                plt.show()
                logging.info(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    # Environment parameters
    map_size = 11 # 11 x 11 grid
    cfg = config.eat.get_config(map_size)
    env = magent2.GridWorld(cfg, render_mode='human')
    env.set_render_dir("eat")
    N = 20 # number of agents
    L = 12 # number of adversaries
    handles = env.get_handles()
    action_size = env.get_action_space(handles[0])[0]
    state_size = 12
    # Hyperparameters
    buffer_size = 200000
    batch_size = 10
    episodes = 2000
    episodes_before_train = 200
    max_steps = 120 # 300 for battle, 120 for jungle
    tau = 0.01
    learning_rate = 10e-4
    epsilon = 0.6 #epsilon in paper, probability that we select a random Q-value
    gamma = 0.96 # for jungle/battle gamma = 0.96, for routing gamma = 0.98, discount
    policy = None

    # Create instances
    replay_buffer = ReplayBuffer(buffer_size)
    agent = Agent(state_size, action_size, learning_rate, tau)
    #model_path = 'model_episode_2000.pth'
    #agent.model.load_state_dict(torch.load(model_path))
    #agent.model.eval()  # Set the model to evaluation mode
    environment = Environment(agent,policy, replay_buffer, env, batch_size, max_steps, tau,
                              epsilon, gamma, N, L)

    # Train the agent
    environment.train(episodes)

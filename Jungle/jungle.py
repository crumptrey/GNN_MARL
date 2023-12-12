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
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

'''
Utilities for converting to graph data
'''


# Function to convert edge index to networkx graph for visualization
def to_networkx(edge_index, node_features):
    G = nx.Graph()
    for i, feature in enumerate(node_features):
        G.add_node(i, feature=feature.tolist())
    for src, dst in edge_index.t().tolist():
        G.add_edge(src, dst)
    return G


# Function to visualize the graph
def visualize_graph(edge_index, node_features):
    G = to_networkx(edge_index, node_features)
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.show()


# Function to print graph statistics
def print_graph_statistics(edge_index, node_features):
    G = to_networkx(edge_index, node_features)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Degree distribution:", [val for (node, val) in G.degree()])


# Function to check adjacency matrix
def check_adjacency_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for src, dst in edge_index.t().tolist():
        adj_matrix[src][dst] = 1
    print("Adjacency Matrix:\n", adj_matrix)


# Function to verify node features
def verify_node_features(node_features):
    print("Node features:\n", node_features)

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
            ((state1[j][0:11, 0:11, 1] - state1[j][0:11, 0:11, 4]).flatten(), state2[j][-1:-3:-1])))
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


def calculate_kl_divergence(weights1, weights2):
    kl_div = F.kl_div(F.log_softmax(weights1, dim=1), F.softmax(weights2, dim=1),
                      reduction='batchmean')
    return kl_div


class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()
        self.fc1 = nn.Linear(123, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def scaled_dot_product_attention(q, k, v, tau, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk
    scaled_attention_logits = matmul_qk / tau

    # Add the mask to the scaled tensor (if applicable)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # Softmax is applied to the last axis (seq_len_k) so that the scores add up to 1
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, tau):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.tau = tau

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = 1
        q = self.split_into_heads(self.wq(q), batch_size)
        k = self.split_into_heads(self.wk(k), batch_size)
        v = self.split_into_heads(self.wv(v), batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, self.tau, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x1, x2, x3):
        #print(x1.size())
        #print(x2.size())
        #print(x3.size())
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DGN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DGN, self).__init__()
        self.encoder = ObservationEncoder()
        self.attention1 = MultiHeadAttention(d_model=128, num_heads=8, tau =0.25)
        self.attention2 = MultiHeadAttention(d_model=128, num_heads=8, tau =0.25)
        self.q_net = QNetwork(input_dim=128 * 3, action_dim=action_dim)

    def forward(self, data):
        # data contains node features and edge indices
        feature = self.encoder(data.x)
        attn_output1, attn_weights1 = self.attention1(feature, feature, feature)  # q, k, v are all the same
        attn_output2, attn_weights2 = self.attention2(attn_output1, attn_output1,
                                       attn_output1)  # q, k, v are all the same
        attn_output1 = torch.squeeze(attn_output1)
        attn_output2 = torch.squeeze(attn_output2)
        q_values = self.q_net(feature, attn_output1, attn_output2)
        return q_values, attn_weights1, attn_weights2


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

class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        # Assuming input_dim is the dimension of the input features
        self.fc1 = nn.Linear(input_dim, 1024)  # First layer
        self.fc2 = nn.Linear(1024, 256)  # Second layer
        self.fc3 = nn.Linear(256, action_dim)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x)  # No activation function for the output layer
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
        action_values, attn_weight1, attn_weight2 = self.model(data)
        return action_values


class IndependentAgent(object):
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.target_model = self.build_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def build_model(self):
        # Define the Q-network architecture here
        model = DQN(input_dim=self.state_size, action_dim=self.action_size)
        model.to(device)
        return model

    def build_target_model(self):
        # Define the Q-network architecture here
        model = DQN(input_dim=self.state_size, action_dim=self.action_size)
        model.to(device)
        return model

    def decide_action(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).to(device)
            action_values = self.model(state)
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
    def __init__(self, agent, policy, replay_buffer, env, batch_size, max_steps, beta, tau,
                 epsilon, gamma, lamb, N, L):
        self.policy = policy
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.env = env
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.beta = beta
        self.tau = tau
        self.epsilon = epsilon
        self.gamma = gamma
        self.lamb = lamb
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

    def learn(self, batch):
        for e in range(5):
            regularize = False
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

            current_q_values, attn_weight1, attn_weight2 = self.agent.model(states)
            current_q_values = current_q_values.view(self.N, len(batch), action_size)

            self.agent.target_model.to(device)
            self.agent.target_model.eval()

            target_q_values, attn_weight_t1, attn_weight_t2 = self.agent.target_model(next_states)
            target_q_values = target_q_values.view(self.N, len(batch), action_size)


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

            # Temporal regularization
            attn_div1 = calculate_kl_divergence(attn_weight1, attn_weight_t1)
            attn_div2 = calculate_kl_divergence(attn_weight2, attn_weight_t2)
            kl_div = self.lamb * 1/2 * (attn_div1 + attn_div2)
            if regularize:
                loss = F.mse_loss(current_q_values, target_q_values) + kl_div
            else:
                loss = F.mse_loss(current_q_values, target_q_values)

            loss.backward()
            q = current_q_values.max().item()
            loss_e = loss.item()
            self.agent.optimizer.step()
            # logging.info(f'Training Loss: {loss.item()}')
            target_net_state_dict = self.agent.target_model.state_dict()
            policy_net_state_dict = self.agent.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.beta + \
                                             target_net_state_dict[
                                                 key] * (1 - self.beta)
            self.agent.target_model.load_state_dict(target_net_state_dict)
        return loss_e, q


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
        steps = 0
        episode_rewards = []
        loss = []
        q_values = []
        score = 0
        while not done and steps < self.max_steps:
            steps += 1
            '''
            Obtain observations of agents and use model to decide on next action for handles[0] group
            '''
            if self.policy == 'DGN':
                observations[0] = self.env.get_observation(handles[0])  # obtain the observations from the N agents
                data = create_graph_data(observations[0][0], observations[0][1])
                state = data.to(device)
                if steps == self.max_steps and episode % 100 == 0:
                    self.env.render()
                    visualize_graph(data.edge_index, data.x)
                    print_graph_statistics(data.edge_index, data.x)
                    check_adjacency_matrix(data.edge_index, data.x.size(0))
                    verify_node_features(data.x)
                acts = self.agent.decide_action(data)
                action[0] = np.zeros(self.N, dtype=np.int32)

            elif self.policy == 'DQN':
                for agent_id in range(self.N):
                    state = observations[agent_id]
                    action = self.agent[agent_id].decide_action(state)
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
            Step to obtain next observation, reward, score
            '''
            next_state, reward, done = self.step(handles)
            score += sum(reward)
            # Add to buffer
            self.replay_buffer.add(state, action[0], reward, next_state, done)
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
                batched_loss, batched_q = self.learn(batch)
                loss.append(batched_loss)
                q_values.append(batched_q)
            # Occasionally render the environment
            #if (episode - 1) % 100 == 0:
            #    self.env.render()
        total_episode_reward = score / 1
        mean_episode_reward = total_episode_reward / self.N  # Mean reward per agent
        mean_loss = sum(loss)/self.max_steps
        mean_q = sum(q_values)/ self.max_steps / self.N

        return mean_episode_reward, mean_loss, mean_q

    def train(self, episodes):
        mean_reward = []
        mean_loss = []
        mean_q = []
        for episode in range(episodes):
            logging.info(f'Starting episode {episode + 1}/{episodes}')
            reward, loss, q = self.run_episode(episode)
            mean_reward.append(reward)
            mean_loss.append(loss)
            mean_q.append(q)
            logging.info(f'Episode {episode + 1} completed')
            logging.info(
                f'Episode {episode + 1} Summary: Mean Reward: {mean_reward[episode]}')
            # Save the model every 100 episodes or at the end of training
            if (episode + 1) % 100 == 0 or episode + 1 == episodes:
                model_save_path = f'model_episode_{episode + 1}.pth'
                torch.save(self.agent.model.state_dict(), model_save_path)
                # Plotting
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.plot(mean_reward)
                plt.xlabel('Episodes')
                plt.ylabel('Mean Reward')
                plt.title(f'Mean Reward up to Episode {episode + 1}')
                # Subplot for mean Q-value
                plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
                plt.plot(mean_q)
                plt.xlabel('Episodes')
                plt.ylabel('Mean Q-Value')
                plt.title(f'Mean Q-Value up to Episode {episode + 1}')

                plt.subplot(1, 3, 3)
                plt.plot(mean_loss)
                plt.xlabel('Episodes')
                plt.ylabel('Mean Loss')
                plt.title(f'Mean Loss up to Episode {episode + 1}')

                plt.tight_layout()
                plt.savefig(f'results_{episode + 1}.png')
                plt.show()

                logging.info(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    # Environment parameters
    map_size = 30  # 11 x 11 grid
    cfg = config.eat.get_config(map_size)
    env = magent2.GridWorld(cfg, render_mode='human')
    env.set_render_dir("eat")
    N = 20  # number of agents
    L = 12  # number of adversaries
    handles = env.get_handles()
    action_size = env.get_action_space(handles[0])[0]
    state_size = 12
    # Hyperparameters
    buffer_size = 200000
    batch_size = 10
    episodes = 2000
    episodes_before_train = 200
    max_steps = 120  # 300 for battle, 120 for jungle
    beta = 0.01 # soft-update for target weights
    tau = 0.25 # weights for attention
    learning_rate = 10e-4
    epsilon = 0.6  # epsilon in paper, probability that we select a random Q-value
    gamma = 0.96  # discount
    lamb = 0.03 # temporal regularization coefficient
    policy = 'DGN'
    independent = False
    if independent:
        agent = [IndependentAgent(state_size, action_size, learning_rate) for _ in range(N)]
    else:
        agent = Agent(state_size, action_size, learning_rate, tau)
    # Create instances
    replay_buffer = ReplayBuffer(buffer_size)

    # model_path = 'model_episode_2000.pth'
    # agent.model.load_state_dict(torch.load(model_path))
    # agent.model.eval()  # Set the model to evaluation mode
    environment = Environment(agent, policy, replay_buffer, env, batch_size, max_steps, beta, tau,
                              epsilon, gamma, lamb, N, L)

    # Train the agent
    environment.train(episodes)
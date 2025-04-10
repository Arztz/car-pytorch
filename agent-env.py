import argparse
import datetime
import flappy_bird_gymnasium
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from ddqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.nn as nn
import os
import matplotlib
import psutil
from gymnasium.vector import AsyncVectorEnv
import graph

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
g = graph.ShowGraph()
torch.set_num_threads(22)
device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.amp.GradScaler(device)

class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.pretrained_model = hyperparameters.get('pretrained_model', None)
        self.num_envs = hyperparameters.get('num_envs', 8)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')

    def make_env(self):
        def thunk():
            return gymnasium.make(self.env_id, render_mode=None, **self.env_make_params)
        return thunk

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.as_tensor(terminations).float().to(device)
    
        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        with torch.amp.autocast(device):
            loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

    def run(self, is_training=True, render=False):
        if is_training:
            log_message = f"Double DQN Enable" if self.enable_double_dqn else "DQN"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message + '\n')
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time
            print(f"Start time: {start_time}")

        env_fns = [self.make_env() for _ in range(self.num_envs)]
        env = AsyncVectorEnv(env_fns)

        num_states = env.single_observation_space.shape[0]
        num_actions = env.single_action_space.n

        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):
                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            states, _ = env.reset()
            states = torch.tensor(states, dtype=torch.float32, device=device)

            episode_rewards = torch.zeros(self.num_envs, device=device)
            terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

            while not terminations.all():
                if is_training and random.random() < epsilon:
                    actions = torch.tensor([env.single_action_space.sample() for _ in range(self.num_envs)],
                                           dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(states)
                        actions = q_values.argmax(dim=1)

                new_states, rewards, dones, _, _ = env.step(actions.cpu().numpy())
                new_states = torch.tensor(new_states, dtype=torch.float32, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                episode_rewards += rewards

                if is_training:
                    for i in range(self.num_envs):
                        memory.append((states[i].detach(), actions[i].detach(), new_states[i].detach(), rewards[i].detach(), dones[i].item()))

                states = new_states
                terminations |= dones.bool()

            reward_per_episode.append(episode_rewards.mean().item())

            if episode % 10 == 0:
                print(f'{datetime.datetime.now().strftime(DATE_FORMAT)} Episode: {episode} Avg reward {episode_rewards.mean().item()}')

            if is_training:
                max_reward = episode_rewards.max().item()
                if max_reward  > best_reward:
                    print(f"Memory used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
                    print(f"New best reward: {max_reward} at episode {episode}")
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_rewards.mean().item()} at episode {episode}\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = max_reward

                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    g.save_graph(reward_per_episode, epsilon_history, file=self.GRAPH_FILE)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)
    agent.run(is_training=args.train, render=not args.train)

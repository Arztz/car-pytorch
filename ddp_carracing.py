import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

# ==== Preprocessing ====
def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized[np.newaxis, :, :]  # shape = (1, 84, 84)
# ==== Actor ====
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # (84x84) -> (20x20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (20x20) -> (9x9)
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 84, 84)
            dummy_output = self.cnn(dummy_input)
            cnn_output_dim = dummy_output.shape[1]
        print(f"cnn_output_dim: {cnn_output_dim}")
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 256),  # ถ้าใช้ 64x64 input เปลี่ยนเป็น 64*6*6
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Tanh()
        )
    def forward(self, x):
        x = x.to(device)

        print(">> Actor input:", x.shape)
        x = self.cnn(x)

        print(">> CNN output:", x.shape)
        return self.fc(x)

# ==== Critic ====
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 84, 84)
            dummy_output = self.cnn(dummy_input)
            cnn_output_dim = dummy_output.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 3, 256),  # concat กับ action (3 dim)
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        features = self.cnn(state)
        x = torch.cat([features, action], dim=1)
        return self.fc(x)

# ==== DDPG Agent ====
class DDPGAgent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.target_actor = Actor().to(device)
        self.target_critic = Critic().to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer(100_000)
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, noise_std=0.1):
        state = torch.FloatTensor(state).unsqueeze(1).to(device)
        print("select_action State SHAPE:", state.shape)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise_std * np.random.randn(3)
        action[0] = np.clip(action[0], -1, 1)  # steering
        action[1:] = np.clip((action[1:] + 1) / 2, 0, 1)  # gas, brake
        return action

    def train(self, batch_size,steps):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)


        print("TRAIN batch state shape:", state.shape)  

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_action[:, 1:] = (next_action[:, 1:] + 1) / 2  # remap
            target_Q = reward + self.gamma * (1 - done) * self.target_critic(next_state, next_action)

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(state)
        actor_actions = torch.cat([
            actor_actions[:, 0:1],
            (actor_actions[:, 1:] + 1) / 2
        ], dim=1)
        actor_loss = -self.critic(state, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==== Main training loop ====
def main():
    env = gym.make("CarRacing-v3", render_mode="human")  # change to None for headless
    agent = DDPGAgent()
    episodes = 2000
    batch_size = 64

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess(obs)
        # state = np.expand_dims(state, axis=0) 
        print("STATE SHAPE:", state.shape)
        total_reward = 0
        shaped_reward = 0
        minus_count = 0
        episode_reward = 0
        for t in range(1000):
            action = agent.select_action(state)
            action[1:] = (action[1:] + 1) / 2  # convert gas/brake to [0, 1]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            print("Next STATE SHAPE:", next_obs.shape)
            shaped_reward = reward
            if shaped_reward < 0:
                minus_count +=1
            else:
                minus_count = 0
            if minus_count > 100:
                terminated = True

            episode_reward += reward

            if episode_reward < -100:  # ✳️ ปรับ threshold ตามที่เหมาะ
                terminated = True
                print(f"🚫 Episode {ep} terminated early at step {t} due to poor reward ({episode_reward:.2f})")
                    

            done = terminated or truncated

            next_state = preprocess(next_obs)
            # next_state = np.expand_dims(next_state, 0) 
            agent.buffer.push(state, action, reward, next_state, done)

            agent.train(batch_size,steps=t)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {ep} | Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()

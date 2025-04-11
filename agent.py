import argparse
import datetime
import cv2
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from cnn import CNN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.nn as nn
import os
import matplotlib
import psutil
from torchvision import transforms
import graph
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
g = graph.ShowGraph()
torch.set_num_threads(22)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),                    # แปลงจาก numpy -> PIL
    transforms.Resize((84, 84)),               # ปรับขนาด
    transforms.Grayscale(),                    # แปลงเป็นภาพขาวดำ
    transforms.ToTensor(),                     # แปลงเป็น tensor และ normalize ให้อยู่ใน [0,1]
])

class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.env_id             = hyperparameters['env_id']
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
        self.env_make_params = hyperparameters.get('env_make_params',{})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.pretrained_model = hyperparameters.get('pretrained_model', None)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.scaler = torch.amp.GradScaler(device=device)

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = [a if a.shape == torch.Size([1]) else torch.tensor([a], dtype=torch.long, device=device) for a in actions]
        actions = torch.stack(actions).to(device)   
        new_states = torch.stack(new_states).to(device) 
        rewards = torch.stack(rewards).to(device)   
        terminations = torch.as_tensor(terminations).float().to(device)
        with torch.amp.autocast(device_type=device):

            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1,index=actions).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def run(self,is_training=True,render=False):

        if is_training:
            log_message = f"DQN\n"

            if self.enable_double_dqn == True:
                log_message = f"Double DQN Enable"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message+ '\n')
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time
            log_message = f"Start time: {start_time}"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(log_message+ '\n')

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make(self.env_id, render_mode='human' if render else 'rgb_array', **self.env_make_params)
        
        num_states = 27648

        num_actions = env.action_space.n
        
        reward_per_episode = []
        epsilon_history = []
        optimize_every_n_steps = 10
        print(f'state: {num_states}  action: {num_actions}')
        policy_dqn = CNN(num_actions,self.enable_dueling_dqn ).to(device)

        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):
                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = CNN(num_actions,self.enable_dueling_dqn  ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -9999999
        else:
            # Load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            print(f"Loaded pretrained model from {self.MODEL_FILE}")
            policy_dqn.eval()

        #  Training loop
        for episode in itertools.count():
            
            state, _ = env.reset()    
            state = image_preprocessing(state)       
            state = torch.as_tensor(state,dtype=torch.float,device=device).unsqueeze(0)

            terminated = False
            episode_reward = 0.0
            landed_bonus_given = False
            max_timesteps = 2000
            steps = 0
            while (not terminated and episode_reward < self.stop_on_reward):
                steps += 1
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.as_tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax(dim=1)

                # Processing:
                # print(f"action: {action}, type: {type(action)}")
                new_state, reward, terminated, _, info = env.step(int(action.item()))
                new_state = image_preprocessing(new_state)

                shaped_reward = reward
                if max_timesteps < steps:
                    # shaped_reward -= 5
                    terminated = True
                # else:
                #     shaped_reward -= 0.01
                # if new_state[3] < 0:  # vel_y < 0 (ลง)
                #     shaped_reward += 0.02

                # # ถ้า lander ยกตัวขึ้น = ลงโทษเล็กน้อย
                # if new_state[3] > 0:  # vel_y > 0 (ขึ้น)
                #     shaped_reward -= 0.2

                # # ถ้าแตะพื้น = ให้รางวัลเยอะๆ
                # if not landed_bonus_given and (new_state[6] > 0.5 or new_state[7] > 0.5):
                #     shaped_reward += 10.0
                #     landed_bonus_given = True
                episode_reward += shaped_reward
                if shaped_reward < 0:
                    minus_count +=1
                else:
                    minus_count = 0
                if minus_count > 100:
                    terminated = True
                if episode_reward < -100:  # ✳️ ปรับ threshold ตามที่เหมาะ
                    terminated = True
                    print(f"🚫 Episode {episode} terminated early at step {steps} due to poor reward ({episode_reward:.2f})")
                    
                new_state = torch.as_tensor(new_state,dtype=torch.float,device=device).unsqueeze(0)
                reward = torch.as_tensor(reward,dtype=torch.float,device=device)
                

                if is_training:
                    memory.append((state.detach(),action.detach() if action.requires_grad else action,new_state.detach(),reward.detach(),terminated))
                    step_count += 1
                    # if step_count % optimize_every_n_steps == 0:
                    #     mini_batch = memory.sample(self.mini_batch_size)
                    #     self.optimize(mini_batch,policy_dqn,target_dqn)

                state = new_state
            if max_timesteps>steps:
                steps = 0
            reward_per_episode.append(episode_reward)
            if episode % 50 == 0:
                print(f'{datetime.datetime.now().strftime(DATE_FORMAT)} Episode: {episode} reward {episode_reward}')
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {episode_reward} at episode {episode}"
                    print(f"Memory used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    g.save_graph(reward_per_episode, epsilon_history,file=self.GRAPH_FILE)
                    last_graph_update_time = current_time
                if len(memory) > self.mini_batch_size:
                                            #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
                    
        env.close() 

def image_preprocessing(img):
    return transform(img).squeeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model')
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameters_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True,render=False)
    else:
        dql.run(is_training=False,render=True)

    agent = Agent('cartpole1')
    # agent.run(is_training=True,render=True)
    agent.run(is_training=False,render=True)
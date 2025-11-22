
import torch.optim as optim
from flow_gan import ConditionalFlow, Discriminator
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import gym
from tqdm import tqdm
import os
from coolname import generate_slug
import json
import d4rl



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_finetune(self, dataset):
        self.ptr = dataset['observations'].shape[0]
        self.size = dataset['observations'].shape[0]
        self.state[:self.ptr] = dataset['observations']
        self.action[:self.ptr] = dataset['actions']
        self.next_state[:self.ptr] = dataset['next_observations']
        self.reward[:self.ptr] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:self.ptr] = 1. - dataset['terminals'].reshape(-1, 1)

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def clip_to_eps(self, eps=1e-5):
        lim = 1 - eps
        self.action = np.clip(self.action, -lim, lim)

def make_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_flow_gan():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=int(2e5))  # GAN 需要更多训练
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--g_lr', type=float, default=1e-4)  # 生成器学习率
    parser.add_argument('--d_lr', type=float, default=1e-4)  # 判别器学习率
    parser.add_argument('--adv_weight', type=float, default=0.01, help="对抗损失的权重")
    parser.add_argument('--work_dir', type=str, default='results/train_flow_gan')
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # --- 设置环境和目录 ---
    set_seed_everywhere(args.seed)
    exp_name = f"{args.env}"
    work_dir = os.path.join(args.work_dir, exp_name)
    make_dir(work_dir)
    with open(os.path.join(work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # --- 加载数据 ---
    env_name = f"{args.env}"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    # D4RL数据集状态未标准化，通常需要标准化
    mean, std = replay_buffer.normalize_states()

    # --- 初始化模型和优化器 ---
    generator = ConditionalFlow(action_dim, state_dim).to(args.device)
    discriminator = Discriminator(state_dim, action_dim).to(args.device)

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_lr) # 优化器，相当于结构体
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_lr)

    adversarial_loss_fn = torch.nn.BCEWithLogitsLoss()      #它被赋值成了一个方法

    # --- 训练循环 ---
    for step in tqdm(range(args.num_iters + 1), desc='Training Flow-GAN'):
        states, actions, _, _, _ = replay_buffer.sample(args.batch_size)

        # === 训练判别器 ===
        optimizer_d.zero_grad()

        # 真实样本
        real_output = discriminator(states, actions)
        loss_d_real = adversarial_loss_fn(real_output, torch.ones_like(real_output)) # 生成一个与他形状相同的全为1的张量

        # 生成的伪样本
        with torch.no_grad():
            fake_actions = generator.sample(args.batch_size, states)
        fake_output = discriminator(states, fake_actions)
        loss_d_fake = adversarial_loss_fn(fake_output, torch.zeros_like(fake_output)) # 生成一个与他形状相同的全为0的张量

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # === 训练生成器 ===
        optimizer_g.zero_grad()

        # 1. NLL 损失 (密度估计)
        loss_g_nll = -generator.log_prob(actions, states).mean()

        # 2. 对抗损失 (欺骗判别器)
        generated_actions = generator.sample(args.batch_size, states)
        d_on_fake = discriminator(states, generated_actions)
        loss_g_adv = adversarial_loss_fn(d_on_fake, torch.ones_like(d_on_fake))

        loss_g = loss_g_nll + args.adv_weight * loss_g_adv
        loss_g.backward()
        optimizer_g.step()

        if step % 5000 == 0:
            print(
                f"\nStep: {step}, D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}, NLL: {loss_g_nll.item():.4f}")
            torch.save(generator.state_dict(), os.path.join(work_dir, f'flow_model_{step}.pt'))


if __name__ == "__main__":
    train_flow_gan()
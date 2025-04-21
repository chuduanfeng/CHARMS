import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from stable_baselines3 import DQN
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


env = gym.make("level2-v0", render_mode=None)
model = DQN.load("/whut_data/wjy/CHARMS/dqn_checkpoints/level2_saal/level2_saal_500000_steps.zip", env=env)

# human data
expert_observations = np.load('bc_data/observations_balanced.npy')
expert_q_values = np.load('bc_data/actions_balanced.npy')

class HighDDataset(Dataset):
    def __init__(self, observations, q_values):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.q_values[idx]

dataset = HighDDataset(expert_observations, expert_q_values)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

q_net = model.policy.q_net
q_net.train()
optimizer = optim.Adam(q_net.parameters(), lr=1e-6)

# regularization
initial_params = [p.clone().detach() for p in q_net.parameters()]
lambda_reg = 0.2
loss_fn = nn.KLDivLoss(reduction='batchmean')

num_epochs = 8
loss_list = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_obs, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_obs = batch_obs.to(model.device)
        batch_labels = batch_labels.to(model.device)

        optimizer.zero_grad()

        logits = q_net(batch_obs)  # output shape [B, num_actions]
        log_probs = torch.log_softmax(logits, dim=-1)

        imitation_loss = loss_fn(log_probs, batch_labels)
        reg_loss = sum((p - p0.to(p.device)).pow(2).sum() for p, p0 in zip(q_net.parameters(), initial_params))

        total_loss = imitation_loss + lambda_reg * reg_loss
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() * batch_obs.size(0)

    epoch_loss /= len(dataset)
    loss_list.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_loss:.4f}")

# update target network
model.policy.q_net.load_state_dict(q_net.state_dict())
model.policy.q_net_target.load_state_dict(q_net.state_dict())
model.save("/whut_data/wjy/CHARMS/dqn_checkpoints/level2_saal/level2_saal_with_bc.zip")
print("\nLoss list over epochs:", loss_list)

import os
import sys
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from stable_baselines3 import DQN
import highway_env
from tqdm import trange


# Create the environment
env = gym.make("test2-v0", render_mode="rgb_array")
env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.Level2Vehicle"
env.unwrapped.config["training_level"] = 3

speed_dhw_pairs = []
lane_change_dhw = []
lane_change_flags = {}

for episode in trange(1800, desc="Episode"):
    done = truncated = False
    obs, info = env.reset()
    for idx, vehicle in enumerate(env.road.vehicles):
        lane_change_flags[idx] = False
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(None)
        for idx, vehicle in enumerate(env.road.vehicles):
            front_vehicle, _ = env.road.neighbour_vehicles(vehicle, vehicle.lane_index)
            if front_vehicle and front_vehicle.position[0] - vehicle.position[0] < 50:
                dhw = front_vehicle.position[0] - vehicle.position[0]
                speed_dhw_pairs.append((vehicle.speed, dhw))
                # 检测换道
                if vehicle.lane_index != vehicle.target_lane_index and not lane_change_flags[idx]:
                    lane_change_flags[idx] = True
                    dhw = front_vehicle.position[0] - vehicle.position[0]
                    lane_change_dhw.append(dhw)
            if vehicle.lane_index == vehicle.target_lane_index and lane_change_flags[idx]:
                lane_change_flags[idx] = False
env.close()

# 存储数据
np.save('speed_ttc_CHARMS.npy', speed_dhw_pairs)
np.save('lane_change_dhw_CHARMS.npy', lane_change_dhw)

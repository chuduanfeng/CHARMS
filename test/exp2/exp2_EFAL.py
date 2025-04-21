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
env = gym.make("level2-v0", render_mode="rgb_array")

steps = 0
lane_change = 0
acc_list = []
steer_list = []
speed_list = []
dhw_list = []

ACC_EXTREME_THRESHOLD = 2.5
STEER_EXTREME_THRESHOLD = 0.25
model = DQN.load("dqn_checkpoints/level2_efal/level2_efal_with_bc", env=env) # efco efeg efpr efal saco saeg sapr saal
for episode in trange(1800, desc="Episode"):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        front_vehicle, rear_vehicle = env.vehicle.road.neighbour_vehicles(
            env.vehicle, env.vehicle.lane_index
        )
        if front_vehicle and front_vehicle.position[0] - env.vehicle.position[0] < 50:
            steps += 1
            # lane change
            if action == 0 or action == 2:
                lane_change += 1
            # dhw
            front_vehicle, rear_vehicle = env.vehicle.road.neighbour_vehicles(
                env.vehicle, env.vehicle.lane_index
            )
            if front_vehicle:
                dhw_list.append(front_vehicle.position[0] - env.vehicle.position[0])
            # acc
            acc_list.append(env.vehicle.action['acceleration'])
            # steering
            steer_list.append(env.vehicle.action['steering'])
            # speed
            speed_list.append(env.vehicle.speed)
env.close()
acc_mean = np.mean([acc for acc in acc_list if acc > 0]) if [acc for acc in acc_list if acc > 0] else 0
acc_std = np.std(acc_list) if acc_list else 0.0
speed_mean = np.mean(speed_list) if speed_list else 0.0
speed_std = np.std(speed_list) if speed_list else 0.0
dhw_mean = np.mean(dhw_list) if dhw_list else 0.0
dhw_std = np.std(dhw_list) if dhw_list else 0.0
lane_change_freq = lane_change / steps
acc_extreme_frequency = np.sum(np.abs(acc_list) > ACC_EXTREME_THRESHOLD) / len(acc_list)
steer_extreme_frequency = np.sum(np.abs(steer_list) > STEER_EXTREME_THRESHOLD) / len(steer_list)

print("Speed mean: {:.3f}, std: {:.3f}".format(speed_mean, speed_std))
print("Acceleration mean: {:.3f}, std: {:.3f}".format(acc_mean, acc_std))
print("Distance Headway mean: {:.3f}, std: {:.3f}".format(dhw_mean, dhw_std))
print("Lane change frequency: {:.3f}".format(lane_change_freq))
print("Acceleration extreme frequency: {:.3f}".format(acc_extreme_frequency))
print("Steering extreme frequency: {:.3f}".format(steer_extreme_frequency))

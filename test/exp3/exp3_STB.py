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
env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.StackelbergLateralVehicle"

steps = 0
acc_list = []
steer_list = []
speed_list = []
ttc_list = []
env_speed = []
env_acc = []
interact_list = []

ACC_EXTREME_THRESHOLD = 2.5
STEER_EXTREME_THRESHOLD = 0.25

for episode in trange(1800, desc="Episode"):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(None)
        front_vehicle, rear_vehicle = env.vehicle.road.neighbour_vehicles(
            env.vehicle, env.vehicle.lane_index
        )
        if front_vehicle and front_vehicle.position[0] - env.vehicle.position[0] < 50:
            steps += 1
            # 遍历所有车辆
            interaction = 0
            for vehicle in env.road.vehicles:
                env_speed.append(vehicle.speed)
                env_acc.append(vehicle.action["steering"])
                if abs(vehicle.position[0] - env.vehicle.position[0]) < 30 and abs(vehicle.lane_index[2] - env.vehicle.lane_index[2]) <= 1:
                    interaction += 1
            interact_list.append(interaction)
            # ttc
            front_vehicle, rear_vehicle = env.road.neighbour_vehicles(
                env.vehicle, env.vehicle.lane_index
            )
            if front_vehicle and (env.vehicle.speed - front_vehicle.speed) > 0:
                ttc = (front_vehicle.position[0] - env.vehicle.position[0]) / (env.vehicle.speed - front_vehicle.speed)
                if ttc > 100:
                    ttc = 100
                ttc_list.append(ttc)
            # acc
            acc_list.append(env.vehicle.action['acceleration'])
            # steering
            steer_list.append(env.vehicle.action['steering'])
            # speed
            speed_list.append(env.vehicle.speed)
env.close()

speed_mean = np.mean(speed_list) if speed_list else 0.0
speed_std = np.std(speed_list) if speed_list else 0.0
danger_ttc = np.sum(np.array(ttc_list) < 3) / len(ttc_list)
env_speed_mean = np.mean(env_speed) if env_speed else 0.0
env_speed_std = np.std(env_speed) if env_speed else 0.0
env_acc_mean = np.mean(env_acc) if env_acc else 0.0
env_acc_std = np.std(env_acc) if env_acc else 0.0
interact_mean = np.mean(interact_list) if interact_list else 0.0
acc_extreme_frequency = np.sum(np.abs(acc_list) > ACC_EXTREME_THRESHOLD) / steps
steer_extreme_frequency = np.sum(np.abs(steer_list) > STEER_EXTREME_THRESHOLD) / steps

print("All vehicles Speed mean: {:.3f}, std: {:.3f}".format(env_speed_mean, env_speed_std))
print("All vehicles Acceleration mean: {:.3f}, std: {:.3f}".format(env_acc_mean, env_acc_std))
print("Ego interaction mean: {:.3f}".format(interact_mean))
print("danger ttc rate: {:.3f}".format(danger_ttc))
print("Ego Speed mean: {:.3f}, std: {:.3f}".format(speed_mean, speed_std))
print("Acceleration extreme frequency: {:.3f}".format(acc_extreme_frequency))
print("Steering extreme frequency: {:.3f}".format(steer_extreme_frequency))

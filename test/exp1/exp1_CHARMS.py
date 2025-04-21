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
env = gym.make("test1-v0", render_mode="rgb_array")

steps = 0
speed_list = []
ttc_list = []
acc_list = []
steer_list = []
collision = 0

ACC_EXTREME_THRESHOLD = 2.5
STEER_EXTREME_THRESHOLD = 0.25

model = DQN.load("dqn_checkpoints/level2_efeg/level2_efeg_with_bc", env=env) # efco efeg efpr efal saco saeg sapr saal
for episode in trange(1800, desc="Episode"):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        front_vehicle, rear_vehicle = env.vehicle.road.neighbour_vehicles(
            env.vehicle, env.vehicle.lane_index
        )
        if front_vehicle and front_vehicle.position[0] - env.vehicle.position[0] < 50:
            # ttc
            if front_vehicle is not None and (env.vehicle.speed - front_vehicle.speed) > 0:
                ttc = (front_vehicle.position[0] - env.vehicle.position[0]) / (env.vehicle.speed - front_vehicle.speed)
                if ttc > 100:
                    ttc = 100
            else:
                ttc = 100
            ttc_list.append(ttc)
            # speed
            speed_list.append(env.vehicle.speed)
            # acc
            acc_list.append(env.vehicle.action['acceleration'])
            # steering
            steer_list.append(env.vehicle.action['steering'])
    if truncated is not True:
        # collision
        collision += 1
        steps += 1
env.close()
average_speed = np.mean(speed_list)
danger_ttc = np.sum(np.array(ttc_list) < 3) / len(ttc_list)
collision_rate = collision / steps
acc_std = np.std(acc_list)
steer_std = np.std(steer_list)
acc_extreme_frequency = np.sum(np.abs(acc_list) > ACC_EXTREME_THRESHOLD) / len(acc_list)
steer_extreme_frequency = np.sum(np.abs(steer_list) > STEER_EXTREME_THRESHOLD) / len(steer_list)

print("collision rate: {:.3f}".format(collision_rate))
print("danger ttc rate: {:.3f}".format(danger_ttc))
print("average speed: {:.2f}".format(average_speed))
print("Acceleration standard deviation: {:.3f}".format(acc_std))
print("Steering standard deviation: {:.3f}".format(steer_std))
print("Acceleration extreme frequency: {:.3f}".format(acc_extreme_frequency))
print("Steering extreme frequency: {:.3f}".format(steer_extreme_frequency))

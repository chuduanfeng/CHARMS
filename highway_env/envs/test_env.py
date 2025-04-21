from typing import Dict, Text

import numpy as np
import random

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork

Observation = np.ndarray


class Test1Env(AbstractEnv):
    """
    A highway driving environment.
    Models as Ego Vehicles.
    """
    ACTIONS: Dict[int, str] = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 3,
                "vehicles_count": 20,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 1,
                "vehicles_density": 1,
                "efficiency_reward": 1.2, # aggressive
                "safety_reward": 0.6,
                "comfort_reward": 0.2,
                "aggressiveness": 1.5,
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "center_vehicles_type": "highway_env.vehicle.behavior.Level2Vehicle",
                "training_level": 0, # 0: use default BVs
            }
        )
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=33
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None: # no-styles
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        center_vehicles_type = utils.class_from_path(self.config["center_vehicles_type"])

        self.controlled_vehicles = []

        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ego_num = np.random.choice(np.arange(0, 6), replace=False)
        for num in numbers:
            if num == ego_num:
                vehicle = center_vehicles_type.create_random(
                    self.road,
                    speed=30,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        self.road.vehicles[0], self.road.vehicles[ego_num] = self.road.vehicles[ego_num], self.road.vehicles[0]

    def acceleration(self, ego_vehicle, front_vehicle):
        acceleration = 3 * (1 - np.power(ego_vehicle.speed / 33, 4))
        desired_gap = 10 + ego_vehicle.speed * 1 + ego_vehicle.speed * (ego_vehicle.speed - front_vehicle.speed) / (2 * np.sqrt(15))
        d = front_vehicle.position[1] - ego_vehicle.position[1]
        acceleration -= 3 * np.power(
            desired_gap / utils.not_zero(d), 2
        )
        return acceleration
    
    def _reward(self, action: Action) -> float:
        """
        The reward includes safety, efficiency, comfort and collision-avoiding.
        :param action: the last action performed
        :return: the corresponding reward
        """
        self_rewards = self._rewards(action)
        self_reward = sum(
            self.config.get(name, 0) * reward for name, reward in self_rewards.items()
        )
        if self.config["normalize_reward"]:
            self_reward = utils.lmap(
                self_reward,
                [0, 2],
                [0, 1],
            )
        if not self.vehicle.on_road:
            self_reward = -1
        if self.vehicle.crashed:
            self_reward = -1
        current_front, current_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        target_front, target_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        if not current_rear:
            current_reward = 0
        elif not current_front:
            current_reward = self.acceleration(current_rear, self.vehicle) - 3 * (1 - np.power(current_rear.speed / 33, 4))
        else:
            current_rear_a = self.acceleration(current_rear, self.vehicle)
            current_rear_pred_a = self.acceleration(current_rear, current_front)
            current_reward = current_rear_pred_a - current_rear_a
        if not target_rear:
            target_reward = 0
        elif not target_front:
            target_reward = self.acceleration(target_rear, self.vehicle) - 3 * (1 - np.power(target_rear.speed / 33, 4))
        else:
            target_rear_a = self.acceleration(target_rear, self.vehicle)
            target_rear_pred_a = self.acceleration(target_rear, target_front)
            target_reward = target_rear_pred_a - target_rear_a
        
        others_reward = np.clip(current_reward, -3, 3) + np.clip(target_reward, -3, 3)
        if self.config["normalize_reward"]:
            others_reward = utils.lmap(
                others_reward,
                [-6, 6],
                [0, 1],
            )
        reward = self.config["aggressiveness"] * self_reward + (1 - self.config["aggressiveness"]) * others_reward

        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        efficiency_reward = np.clip(scaled_speed, 0, 1)

        if action == self.vehicle.last_action:
            comfort_reward = 1
        else:
            comfort_reward = 0

        if action == 0 or action == 2:
            min_rear_distance = 999
            min_front_distance = 999
            closest_front, closest_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
            if closest_front:
                min_front_distance = closest_front.position[0] - self.vehicle.position[0]
            if closest_rear:
                min_rear_distance = self.vehicle.position[0] - closest_rear.position[0]
            if closest_rear and (closest_rear.speed - self.vehicle.speed)>0:
                rear_reward = min_rear_distance / (closest_rear.speed - self.vehicle.speed + 0.01) / 3
                rear_reward = np.clip(rear_reward, 0, 1)
            else:
                rear_reward = 1
            if closest_front and (self.vehicle.speed - closest_front.speed)>0:
                front_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
                front_reward = np.clip(rear_reward, 0, 1)
            else:
                front_reward = 1
            safety_reward = 0.5 * rear_reward + 0.5 * front_reward
        else:
            min_front_distance = 999
            closest_front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if closest_front:
                min_front_distance = closest_front.position[0] - self.vehicle.position[0]
            if closest_front and (self.vehicle.speed - closest_front.speed)>0:
                safety_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
                safety_reward = np.clip(safety_reward, 0, 1)
            else:
                safety_reward = 1

        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward
        }

    def step(self, action: Action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, 0, terminated, truncated, info
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
class Test2Env(AbstractEnv):
    """
    A highway driving environment.
    Models as Environment Vehicles.
    """
    ACTIONS: Dict[int, str] = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 3,
                "vehicles_count": 20,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 1,
                "vehicles_density": 1.5,
                "efficiency_reward": 1.2, # aggressive
                "safety_reward": 0.6,
                "comfort_reward": 0.2,
                "aggressiveness": 1.5,
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "center_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "training_level": 0, # 0: use default BVs 3: use model BVs
            }
        )
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=33
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None: # no-styles
        """Create some new random vehicles of a given type, and add them on the road."""
        if self.config["other_vehicles_type"] == "highway_env.vehicle.behavior.Level2Vehicle":
            # 0: CO, 1: EG, 2: PR, 3: AL
            suffix_map = {0: "CO", 1: "EG", 2: "PR", 3: "AL"}
            ef_list, sa_list = [], []

            while len(ef_list) <= 11 or len(sa_list) <= 10:
                val = np.random.poisson(lam=1.5)
                val = min(val, 3)
                suffix = suffix_map[val]

                if len(ef_list) <= 11:
                    ef_list.append(f"highway_env.vehicle.behavior.Level2EF{suffix}")
                if len(sa_list) <= 10:
                    sa_list.append(f"highway_env.vehicle.behavior.Level2SA{suffix}")
            vehicle_list = ef_list + sa_list
            random.shuffle(vehicle_list)
        elif self.config["other_vehicles_type"] == "BaselineVehicle":
            ef_vehicles = ["highway_env.vehicle.behavior.Level2EFEG"] * 11
            sa_vehicles = ["highway_env.vehicle.behavior.Level2SAEG"] * 10
            vehicle_list = ef_vehicles + sa_vehicles
            random.shuffle(vehicle_list)
        else:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        center_vehicles_type = utils.class_from_path(self.config["center_vehicles_type"])

        self.controlled_vehicles = []

        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        ego_num = np.random.choice(np.arange(0, 6), replace=False)
        for num in numbers:
            if num == ego_num:
                vehicle = center_vehicles_type.create_random(
                    self.road,
                    speed=30,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                if self.config["other_vehicles_type"] == "highway_env.vehicle.behavior.Level2Vehicle" or self.config["other_vehicles_type"] == "BaselineVehicle":
                    other_vehicles_type = utils.class_from_path(vehicle_list[num])
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        self.road.vehicles[0], self.road.vehicles[ego_num] = self.road.vehicles[ego_num], self.road.vehicles[0]

    def acceleration(self, ego_vehicle, front_vehicle):
        acceleration = 3 * (1 - np.power(ego_vehicle.speed / 33, 4))
        desired_gap = 10 + ego_vehicle.speed * 1 + ego_vehicle.speed * (ego_vehicle.speed - front_vehicle.speed) / (2 * np.sqrt(15))
        d = front_vehicle.position[1] - ego_vehicle.position[1]
        acceleration -= 3 * np.power(
            desired_gap / utils.not_zero(d), 2
        )
        return acceleration
    
    def _reward(self, action: Action) -> float:
        """
        The reward includes safety, efficiency, comfort and collision-avoiding.
        :param action: the last action performed
        :return: the corresponding reward
        """
        self_rewards = self._rewards(action)
        self_reward = sum(
            self.config.get(name, 0) * reward for name, reward in self_rewards.items()
        )
        if self.config["normalize_reward"]:
            self_reward = utils.lmap(
                self_reward,
                [0, 2],
                [0, 1],
            )
        if not self.vehicle.on_road:
            self_reward = -1
        if self.vehicle.crashed:
            self_reward = -1
        current_front, current_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        target_front, target_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        if not current_rear:
            current_reward = 0
        elif not current_front:
            current_reward = self.acceleration(current_rear, self.vehicle) - 3 * (1 - np.power(current_rear.speed / 33, 4))
        else:
            current_rear_a = self.acceleration(current_rear, self.vehicle)
            current_rear_pred_a = self.acceleration(current_rear, current_front)
            current_reward = current_rear_pred_a - current_rear_a
        if not target_rear:
            target_reward = 0
        elif not target_front:
            target_reward = self.acceleration(target_rear, self.vehicle) - 3 * (1 - np.power(target_rear.speed / 33, 4))
        else:
            target_rear_a = self.acceleration(target_rear, self.vehicle)
            target_rear_pred_a = self.acceleration(target_rear, target_front)
            target_reward = target_rear_pred_a - target_rear_a
        
        others_reward = np.clip(current_reward, -3, 3) + np.clip(target_reward, -3, 3)
        if self.config["normalize_reward"]:
            others_reward = utils.lmap(
                others_reward,
                [-6, 6],
                [0, 1],
            )
        reward = self.config["aggressiveness"] * self_reward + (1 - self.config["aggressiveness"]) * others_reward

        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        efficiency_reward = np.clip(scaled_speed, 0, 1)

        if action == self.vehicle.last_action:
            comfort_reward = 1
        else:
            comfort_reward = 0

        if action == 0 or action == 2:
            min_rear_distance = 999
            min_front_distance = 999
            closest_front, closest_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
            if closest_front:
                min_front_distance = closest_front.position[0] - self.vehicle.position[0]
            if closest_rear:
                min_rear_distance = self.vehicle.position[0] - closest_rear.position[0]
            if closest_rear and (closest_rear.speed - self.vehicle.speed)>0:
                rear_reward = min_rear_distance / (closest_rear.speed - self.vehicle.speed + 0.01) / 3
                rear_reward = np.clip(rear_reward, 0, 1)
            else:
                rear_reward = 1
            if closest_front and (self.vehicle.speed - closest_front.speed)>0:
                front_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
                front_reward = np.clip(rear_reward, 0, 1)
            else:
                front_reward = 1
            safety_reward = 0.5 * rear_reward + 0.5 * front_reward
        else:
            min_front_distance = 999
            closest_front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if closest_front:
                min_front_distance = closest_front.position[0] - self.vehicle.position[0]
            if closest_front and (self.vehicle.speed - closest_front.speed)>0:
                safety_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
                safety_reward = np.clip(safety_reward, 0, 1)
            else:
                safety_reward = 1

        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward
        }

    def step(self, action: Action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, 0, terminated, truncated, info
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
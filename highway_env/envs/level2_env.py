from typing import Dict, Text
import math
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import Level2Vehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class Level2Env(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
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
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 1,
                "vehicles_density": 1,
                "efficiency_reward": 1.0, # efficiency-prioritized(1.0,0.8,0.2) safety-prioritized(0.8,1.0,0.2)
                "safety_reward": 0.8,
                "comfort_reward": 0.2,
                "svo": -math.pi/4, #competitive:-pi/4 egoistic:0 prosocial:pi/4 altruistic:pi/2
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.Level1Vehicle",
                "training_level": 2
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

    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

    #     self.controlled_vehicles = []
    #     vehicle = Vehicle.create_specific(
    #         self.road, x=220, speed=30, lane_id=1
    #     )
    #     vehicle = self.action_type.vehicle_class(
    #         self.road, vehicle.position, vehicle.heading, vehicle.speed
    #     )
    #     self.controlled_vehicles.append(vehicle)
    #     self.road.vehicles.append(vehicle)
    #     ov_coordinates = [(260, 2, 23), (358, 2, 26), (416, 2, 26),
    #                         (241, 1, 25), (312, 1, 22), (403, 1, 23), 
    #                         (210, 0, 22), (274, 0, 22), (374, 0, 22), (392, 0, 24)]
    #     for x, y, z in ov_coordinates:
    #         vehicle = other_vehicles_type.create_specific(
    #             self.road, x=x, speed=z, lane_id=y
    #         )
    #         self.road.vehicles.append(vehicle)
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []

        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        ego_num = np.random.choice(np.arange(0, 15), replace=False)
        for num in numbers:
            if num == ego_num:
                vehicle = Level2Vehicle.create_random(
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
        The reward includes safety, efficiency, comfort and env-effect.
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
        reward = math.cos(self.config["svo"]) * self_reward + math.sin(self.config["svo"]) * others_reward
        if not self.vehicle.on_road:
            reward = -10
        if self.vehicle.crashed:
            reward = -10
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        efficiency_reward = np.clip(scaled_speed, 0, 1)

        if self.vehicle.last_action == self.vehicle.action:
            comfort_reward = 1
        else:
            comfort_reward = 0
        self.vehicle.last_action = self.vehicle.action

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
                front_reward = np.clip(front_reward, 0, 1)
            else:
                front_reward = 1
            safety_reward = 0.5 * rear_reward + 0.5 * front_reward
            if min_rear_distance < 5 or min_front_distance < 5:
                safety_reward = 0
        else:
            min_front_distance = 999
            closest_front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if closest_front:
                min_front_distance = closest_front.position[0] - self.vehicle.position[0]
            if closest_front and (self.vehicle.speed - closest_front.speed)>0:
                safety_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
                safety_reward = np.clip(safety_reward, 0, 1)
            elif closest_front and min_front_distance < 50:
                safety_reward = min_front_distance / 50.0
                safety_reward = np.clip(safety_reward, 0, 1)
            else:
                safety_reward = 1

        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward
        }

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

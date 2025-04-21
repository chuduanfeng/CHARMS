import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from highway_env import utils
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

data_dir = 'human_data/'
out_dir = 'bc_data/'
observation_data = []
action_data = []
total_vehicle_count = 0

# Group files by prefix
file_groups = defaultdict(dict)
for filename in os.listdir(data_dir):
    if filename.endswith('.csv') and '_' in filename:
        xx, file_type = filename.split('_', 1)
        file_groups[xx][file_type] = os.path.join(data_dir, filename)

for xx, files in tqdm(file_groups.items(), desc="Processing files"):
    recording_meta_path = files.get('recordingMeta.csv')
    y_offset = 0
    if recording_meta_path and os.path.exists(recording_meta_path):
        try:
            df = pd.read_csv(recording_meta_path, nrows=1)
            lower_lane_markings = df['lowerLaneMarkings'].iloc[0]
            markings = [float(x) for x in lower_lane_markings.split(';') if x.strip() != '']
            if len(markings) != 4:
                raise ValueError(f"{xx}: Expected 4 lane marking values, got {len(markings)}")
            y_offset = 4 - (markings[1] + markings[2]) / 2
        except Exception as e:
            print(f"Error processing {recording_meta_path}: {e}")
            continue

    tracks_meta_path = files.get('tracksMeta.csv')
    if tracks_meta_path and os.path.exists(tracks_meta_path):
        try:
            tracks_meta_df = pd.read_csv(tracks_meta_path)
            filtered_ids = tracks_meta_df[
                (tracks_meta_df['drivingDirection'] == 2) &
                (tracks_meta_df['class'] == "Car") &
                (tracks_meta_df['maxXVelocity'] < 33.33)
            ]['id'].tolist()
            total_vehicle_count += len(filtered_ids)
        except Exception as e:
            print(f"Error processing {tracks_meta_path}: {e}")
            continue
    else:
        continue

    tracks_path = files.get('tracks.csv')
    if not tracks_path or not os.path.exists(tracks_path):
        continue

    try:
        tracks_df = pd.read_csv(tracks_path)
    except Exception as e:
        print(f"Error loading tracks file for {xx}: {e}")
        continue

    for obj_id in filtered_ids:
        obj_frames = tracks_df[tracks_df['id'] == obj_id]

        for _, row in obj_frames.iterrows():
            frame = row['frame']

            related_ids = {
                'preceding': row['precedingId'],
                'following': row['followingId'],
                'leftPre': row['leftPrecedingId'],
                'leftAlong': row['leftAlongsideId'],
                'leftFollow': row['leftFollowingId'],
                'rightPre': row['rightPrecedingId'],
                'rightAlong': row['rightAlongsideId'],
                'rightFollow': row['rightFollowingId']
            }

            ego = np.array([
                1,
                row['x'],
                row['y'] + y_offset,
                row['xVelocity'],
                row['yVelocity']
            ])
            state_array = np.zeros((7, 5))
            state_array[0] = ego

            def get_vehicle_state(vid):
                if vid == 0:
                    return np.zeros(5)
                target = tracks_df[(tracks_df['id'] == vid) & (tracks_df['frame'] == frame)]
                if not target.empty:
                    t = target.iloc[0]
                    return np.array([
                        1,
                        t['x'],
                        t['y'] + y_offset,
                        t['xVelocity'],
                        t['yVelocity']
                    ])
                else:
                    return np.zeros(5)

            left_along_state = get_vehicle_state(related_ids['leftAlong'])
            if left_along_state[0] == 1:
                if left_along_state[1] > ego[1]:
                    state_array[1] = left_along_state
                else:
                    state_array[2] = left_along_state
            state_array[1] = state_array[1] if state_array[1][0] else get_vehicle_state(related_ids['leftPre'])
            state_array[2] = state_array[2] if state_array[2][0] else get_vehicle_state(related_ids['leftFollow'])
            state_array[3] = get_vehicle_state(related_ids['preceding'])
            state_array[4] = get_vehicle_state(related_ids['following'])
            right_along_state = get_vehicle_state(related_ids['rightAlong'])
            if right_along_state[0] == 1:
                if right_along_state[1] > ego[1]:
                    state_array[5] = right_along_state
                else:
                    state_array[6] = right_along_state
            state_array[5] = state_array[5] if state_array[5][0] else get_vehicle_state(related_ids['rightPre'])
            state_array[6] = state_array[6] if state_array[6][0] else get_vehicle_state(related_ids['rightFollow'])

            for i in range(1, 7):
                if state_array[i][0]:
                    state_array[i][1] = np.clip(utils.lmap((state_array[i][1] - state_array[0][1]), [-165, 165], [-1, 1]), -1, 1)
                    state_array[i][2] = np.clip(utils.lmap((state_array[i][2] - state_array[0][2]), [-12, 12], [-1, 1]), -1, 1)
                    state_array[i][3] = np.clip(utils.lmap((state_array[i][3] - state_array[0][3]), [-66, 66], [-1, 1]), -1, 1)
                    state_array[i][4] = np.clip(utils.lmap((state_array[i][4] - state_array[0][4]), [-66, 66], [-1, 1]), -1, 1)
            state_array[0][1] = 1
            state_array[0][2] = np.clip(utils.lmap(state_array[0][2], [-12, 12], [-1, 1]), -1, 1)
            state_array[0][3] = np.clip(utils.lmap(state_array[0][3], [-66, 66], [-1, 1]), -1, 1)
            state_array[0][4] = np.clip(utils.lmap(state_array[0][4], [-66, 66], [-1, 1]), -1, 1)
            state_array = state_array.flatten()
            observation_data.append(state_array)

            action = np.zeros(5)
            ax_threshold = 0.2
            vy_threshold = 0.07
            ax = row['xAcceleration']
            vy = row['yVelocity']
            y = row['y'] + y_offset

            def closest_center(y_val):
                return min([0, 4, 8], key=lambda c: abs(y_val - c))

            lane_center = closest_center(y)
            lateral_offset = y - lane_center

            is_changing_left = lateral_offset < 0 and vy < -vy_threshold
            is_changing_right = lateral_offset > 0 and vy > vy_threshold
            is_accelerating = ax > ax_threshold
            is_decelerating = ax < -ax_threshold

            if (is_changing_left or is_changing_right) and (is_accelerating or is_decelerating):
                lane_conf = min(abs(vy / vy_threshold), 1.0)
                accel_conf = min(abs(ax / ax_threshold), 1.0)
                total_conf = lane_conf + accel_conf

                if total_conf > 0:
                    lane_weight = lane_conf / total_conf
                    accel_weight = accel_conf / total_conf

                    if is_changing_left:
                        action[0] = lane_weight
                    elif is_changing_right:
                        action[2] = lane_weight

                    if is_accelerating:
                        action[3] = accel_weight
                    elif is_decelerating:
                        action[4] = accel_weight
            elif is_changing_left:
                action[0] = 1.0
            elif is_changing_right:
                action[2] = 1.0
            elif is_accelerating:
                action[3] = 1.0
            elif is_decelerating:
                action[4] = 1.0
            else:
                action[1] = 1.0
            action_data.append(action)

        print(f"Save complete. Current number of samples: {len(observation_data)}")
        print(f"Total processed vehicles: {total_vehicle_count}")

    observation_array = np.array(observation_data, dtype=np.float32)
    action_array = np.array(action_data, dtype=np.float32)

    np.save(os.path.join(out_dir, "observations.npy"), observation_array)
    np.save(os.path.join(out_dir, "actions.npy"), action_array)

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import highway_env
import math
from stable_baselines3.common.callbacks import CheckpointCallback

TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("level2-v0", render_mode=None)
    env.unwrapped.config["efficiency_reward"] = 1.0
    env.unwrapped.config["safety_reward"] = 0.8
    env.unwrapped.config["comfort_reward"] = 0.2
    env.unwrapped.config["svo"] = -math.pi / 4
    obs, info = env.reset()

    checkpoint_dir = "./dqn_checkpoints/level2_efco/" # efco efeg efpr efal saco saeg sapr saal
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="level2_efco", # efco efeg efpr efal saco saeg sapr saal
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=2,
        gradient_steps=1,
        target_update_interval=500,
        verbose=0,
        tensorboard_log="dqn_checkpoints/log/",
        device="cuda:1"
    )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e6), progress_bar=True, callback=checkpoint_callback)
        model.save("dqn_checkpoints/level2/level2_ef_co_model") # efco efeg efpr efal saco saeg sapr saal
        del model

    # Run the trained model and record video
    model = DQN.load("dqn_checkpoints/level2/level2_ef_co_model", env=env) # efco efeg efpr efal saco saeg sapr saal
    env = RecordVideo(
        env, video_folder="dqn_checkpoints/level2/level2_ef_co_videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.unwrapped.config["simulation_frequency"] = 15

    for videos in range(3):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()


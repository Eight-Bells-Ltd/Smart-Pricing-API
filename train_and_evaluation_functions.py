import functools
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from __future__ import annotations
import glob
import time
import supersuit as ss
import cv2
import sys
import gymnasium
from gymnasium.spaces import MultiDiscrete, Discrete, Dict, Box

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import parallel_to_aec, wrappers

from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy

from reverse_auction_env import ReverseAuctionEnv


def create_video_from_pngs(images_folder):

    # Output video file name
    output_video = "output_video.mp4"

    # Get a list of all PNG files in the folder and their creation times
    image_files_with_dates = [(os.path.join(images_folder, file), os.stat(os.path.join(images_folder, file)).st_mtime) for file in os.listdir(images_folder) if file.endswith(".png")]

    # Sort the files based on creation date
    image_files_sorted = sorted(image_files_with_dates, key=lambda x: x[1])

    # Extract file paths from the sorted list
    image_files = [file[0] for file in image_files_sorted]

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    fps = 0.8  # Decrease the frame rate to increase the delay between frames
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through each image and write it to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        video.write(image)

    # Release the video object
    video.release()

    print(f"Video created successfully: {output_video}")

def train(env_fn, steps: int = 10_000, learning_rate=1e-3, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment
    
    #env = env_fn.parallel_env(**env_kwargs)
    
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MultiInputPolicy,
        env,
        verbose=3,
        learning_rate=learning_rate,
        batch_size=256,
        ent_coef=0.5,  # Adjust the value as needed

    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def evaluate(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)
    env = parallel_to_aec(env)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            
            obs, reward, termination, truncation, info = env.last()
            if not termination or not truncation:
                if agent == env.agents[-1]: env.render()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
            
        if not termination or not truncation:env.render()
            
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


############ This main is for rain and testing the model ###################
if __name__ == "__main__":

    mode = sys.argv[1]
    if mode == "train":
        env = ReverseAuctionEnv()
        parallel_api_test(env, num_cycles=10_000)

        env_fn = ReverseAuctionEnv
        
        env_kwargs = {}

        # Train a model (takes ~3 minutes on GPU)
        train(env_fn, steps=200_000, learning_rate=1e-3 , seed=42, **env_kwargs)

    if mode == "test":
        env_fn = ReverseAuctionEnv
    
        env_kwargs = {}

        evaluate(env_fn, num_games=1, render_mode="human", **env_kwargs)

        images_folder = "outputs"
        create_video_from_pngs(images_folder)



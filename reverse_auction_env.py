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

import gymnasium
from gymnasium.spaces import MultiDiscrete, Discrete, Dict, Box

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import parallel_to_aec, wrappers

from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy



class ReverseAuctionEnv(ParallelEnv):

    metadata = {"render_modes": ["human"], "name": "reverse_auction_v1"}

    def __init__(self, render_mode="human", num_bidders=5, max_rounds=20):
        self.possible_agents = ["provider_" + str(r) for r in range(num_bidders)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.render_mode = render_mode
        self.max_rounds = max_rounds
        self.round = 1
        self.done = False
        self.initial_bids = np.random.randint(55, 60, size=num_bidders).astype(np.float64)        
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(num_bidders, dtype=int)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            'current_rank': Discrete(len(self.possible_agents) + 1),
            'previous_rank': Discrete(len(self.possible_agents) + 1),
            'current_round': Discrete(self.max_rounds + 1)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)  # 0->same bid, 1-4>higher bid, 5-8>lower bid

    
    def render(self):
        if self.render_mode == "human":
            num_agents = len(self.possible_agents)
            round_number = self.round
            plt.figure(figsize=(10, 6))
    
            for i in range(num_agents):
                agent_bid = self.bids[i]
                plt.plot([i + 1], [agent_bid], marker='o', label=f"{self.possible_agents[i]} Bid")
            
            plt.xlabel("Agents")
            plt.ylabel("Bid Value")
            plt.ylim(0, 100)  # Set y-axis boundaries
            plt.title(f"Bids of Each Agent in Round {round_number}")
            plt.legend()
            plt.grid(True)
            
            output_folder = "outputs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
            output_path = os.path.join(output_folder, f"round_{round_number}_bids.png")
            plt.savefig(output_path)
            plt.close()
            
        
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.round = 1
        self.done = False
        self.initial_bids = np.random.randint(55, 60, size=len(self.possible_agents)).astype(np.float64)  
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(len(self.agents), dtype=int)

        rank = 1

        observations = {agent: {'current_rank': rank, 'previous_rank': rank, 'current_round': self.round} for agent in self.agents}
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            #self.agents = []
            return {}, {}, {}, {}, {}

        if self.round == self.max_rounds:
            self.done = True

        rewards = {}
        observations = {}
        
        for agent in self.agents:
            action = actions[agent]
            agent_id = self.agent_name_mapping[agent]

            if action == 0:  # Same bid
                pass
            elif 1 <= action <= 4:  # Higher bid
                self.bids[agent_id] *= 1.1 + (action - 1) * 0.1
            elif 5 <= action <= 8:  # Lower bid
                self.bids[agent_id] *= 0.9 - (action - 5) * 0.1


        sorted_indices = np.argsort(self.bids)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]
            previous_rank = self.prev_ranks[agent_id]
            current_rank = ranks[agent_id]

            rewards[agent] = 0
            
            # Reward for improving rank
            if current_rank < previous_rank:
                rewards[agent] += 1  
            # Negative reward for worsening rank
            elif current_rank > previous_rank:
                rewards[agent] -= 1  

            
            # Negative reward for bid outside range
            if not (30 < self.bids[agent_id] < 60):
                rewards[agent] -= 30  
            else:
                rewards[agent] += 30
            
            # Reward for being first
            if current_rank == 1:
                if (30 < self.bids[agent_id] < 40): rewards[agent] += 3 
                else: rewards[agent] -= 1
            # Negative reward for not being first in the last round
            elif self.round == self.max_rounds and current_rank != 1:
                rewards[agent] -= 1
            
            observations[agent] = {
                'current_rank': current_rank,
                'previous_rank': previous_rank,
                'current_round': self.round
            }
            
            self.prev_ranks[agent_id] = current_rank


        self.state = observations

        infos = {agent: {} for agent in self.agents}
        terminations = {agent: self.done for agent in self.agents}
        truncations = {agent: self.done for agent in self.agents}

        self.round += 1

        if self.done:
            self.agents = [] 

        return observations, rewards, terminations, truncations, infos
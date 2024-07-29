import functools
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box
from pettingzoo import ParallelEnv

from utils.helpers import calculate_reward, update_bid, render_env

import os

import json


class ReverseAuctionEnv(ParallelEnv):
  
    metadata = {"render_modes": ["human"], "name": "reverse_auction_v1"}

    def __init__(self, render_mode="human", num_bidders=5, possible_agents={}, initial_prices={}, avg_min=30, max_rounds=20):
        self.possible_agents = possible_agents if possible_agents != {} else ["provider_" + str(r) for r in range(num_bidders)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.render_mode = render_mode
        self.max_rounds = max_rounds
        self.round = 1
        self.avg_min = avg_min
        self.done = False
        self.initial_bids = initial_prices if initial_prices != {} else np.random.randint(85, 100, size=num_bidders).astype(np.float64)       
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(num_bidders, dtype=int)
        print (possible_agents)

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
            bids = self.bids
            agent_list = self.possible_agents
            output_folder = "outputs/pngs"

            render_env(num_agents, round_number, bids, agent_list, output_folder)
            
        
    def close(self):

        min_index = np.argmin(self.bids)
        winner = self.possible_agents[min_index]
   

        data = {
            "winner": winner,
            "price": np.min(self.bids)
        }

        with open('auction_results.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print("Data successfully written to output.json") 

        return

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.round = 1
        self.done = False
        self.bids = self.initial_bids.copy()
        self.prev_ranks = np.ones(len(self.agents), dtype=int)


        observations = {agent: {'current_rank': 1, 'previous_rank': 1, 'current_round': 1} for agent in self.agents}
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

            self.bids[agent_id] = update_bid(action, self.bids[agent_id])

        sorted_indices = np.argsort(self.bids)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices)) + 1

        for agent in self.agents:
            agent_id = self.agent_name_mapping[agent]
            previous_rank = self.prev_ranks[agent_id]
            current_rank = ranks[agent_id]

            rewards[agent] = calculate_reward(current_rank, previous_rank, self.avg_min, self.bids[agent_id], self.initial_bids[agent_id], self.round, self.max_rounds)
            
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

        if(self.done!=True): self.round += 1

        if self.done:
            self.agents = [] 


        return observations, rewards, terminations, truncations, infos

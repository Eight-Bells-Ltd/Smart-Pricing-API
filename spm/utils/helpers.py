import numpy as np
import json

def get_results(bids, agent_list):
    # Determine the winner and winning price
    winner_data = {"winner": agent_list[np.argmin(bids)], "price": np.min(bids)}
    
    # Write results to a JSON file
    with open('auction_results.json', 'w') as json_file:
        json.dump(winner_data, json_file, indent=4)

    print("Data successfully written to auction_results.json")

def update_bid(action, bid):

    multipliers = [1.0, 1.1, 1.2, 1.3, 1.4, 0.9, 0.8, 0.7, 0.6]

    return bid * multipliers[action] if 0 <= action < len(multipliers) else bid

def calculate_reward(current_rank, previous_rank, avg_min, bid, initial_bid, round, max_rounds):

    reward = 20 if avg_min < bid < initial_bid else -80

    if current_rank < previous_rank: reward += 10

    # reward += 10 if current_rank < previous_rank else -10 if current_rank > previous_rank else 0

    if round == max_rounds: reward += 5 * max_rounds if current_rank == 1 else -10 * max_rounds

    return reward
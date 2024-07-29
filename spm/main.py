import argparse
import yaml
from environment.reverse_auction_env import ReverseAuctionEnv
from training.train import train
from evaluation.evaluate import evaluate
from utils.helpers import create_video_from_pngs

def main():
    parser = argparse.ArgumentParser(description="Reverse Auction Simulation")
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--config', default='config.yml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env_kwargs = config['environment']
    env_fn = ReverseAuctionEnv

    if args.mode == 'train':
        train_config = config['training']
        train(env_fn, steps=train_config['steps'], learning_rate=train_config['learning_rate'], 
              seed=train_config['seed'], batch_size=train_config['batch_size'], ent_coef=train_config['ent_coef'], **env_kwargs)
    elif args.mode == 'evaluate':
        eval_config = config['evaluation']
        evaluate(env_fn, num_games=eval_config['num_games'], 
                 render_mode=eval_config['render_mode'], **env_kwargs)

    # Create video after evaluation
    if args.mode == 'evaluate':
        create_video_from_pngs("outputs/pngs")

if __name__ == "__main__":
    main()
import warnings
import datetime
import logging
import os
from pathlib import Path
import torch
from config_utils import load_or_create_config
from train import initial_setup, train_model
from fetch_data import fetch_oanda_candles_range, add_technical_indicators
from di_model import create_forex_agent, get_default_config
from forex_env import create_di_env
from ding.envs import DingEnvWrapper
from ding.worker.collector import base_serial_evaluator, base_serial_collector

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore")
DATE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

def load_policy(policy_path: str, env):
    """Load a saved DI-engine policy"""
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file {policy_path} not found")
    
    cfg = get_default_config()
    cfg.policy.model.obs_shape = env.observation_space.shape[0]
    cfg.policy.model.action_shape = env.action_space.n
    
    policy = create_forex_agent(env, cfg)
    policy.load_state_dict(torch.load(policy_path))
    return policy

def main():
    config = load_or_create_config()
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--granularity', type=str, required=True, help="Granularity (e.g., 'H1')")
    parser.add_argument('--instrument', type=str, required=True, help="Instrument (e.g., 'EUR_USD')")
    parser.add_argument('--max-profit-percent', type=float, default=0.1)
    parser.add_argument('--max-loss-percent', type=float, default=0.1)
    args = parser.parse_args()

    # Setup dates and paths
    start_date = (datetime.datetime.now() - datetime.timedelta(days=3650)).strftime(DATE_FORMAT)
    end_date = datetime.datetime.now().strftime(DATE_FORMAT)
    model_dir = Path("trained_models")
    model_dir.mkdir(exist_ok=True)
    
    # List available policies
    available_policies = list(model_dir.glob("*.pth"))
    print("Available pre-trained policies:")
    for policy_path in available_policies:
        print(f"- {policy_path.name}")
    
    policy_name = input("Enter policy name to load (leave blank to train new): ").strip()
    
    initial_setup()
    try:
        # Fetch and prepare data
        df = fetch_oanda_candles_range(
            args.instrument, start_date, end_date, 
            args.granularity, config["access_token"]
        )
        df = add_technical_indicators(df)
        print(f"Prepared DataFrame shape: {df.shape}")
        
        # Create environment
        env = create_di_env(df, window_size=10, frame_bound=(10, len(df)))
        env = DingEnvWrapper(env)
        
        if policy_name:
            # Load existing policy
            policy_path = model_dir / policy_name
            policy = load_policy(str(policy_path), env)
            print(f"Loaded policy from {policy_path}")
        else:
            # Train new policy
            print("Training new policy...")
            policy = train_model(df)
            print("Training completed")
        
        # Initialize trading bot with policy
        from trade import TradingBot
        bot = TradingBot(
            policy=policy,
            instrument=args.instrument,
            access_token=config["access_token"],
            max_profit_percent=args.max_profit_percent,
            max_loss_percent=args.max_loss_percent
        )
        
        # Start trading
        should_trade = (policy_name != '') or (
            input("Start trading with new policy? [y/n]: ").lower() == 'y'
        )
        
        if should_trade:
            bot.start_live_trading()
        else:
            print("Exiting without trading...")
            
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

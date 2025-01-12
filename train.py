import os
import logging
from typing import Dict
import json
import hashlib
import torch
import pandas as pd
from forex_env import create_di_env
from di_model import create_forex_agent, get_default_config
from ding.envs import DingEnvWrapper
from ding.worker import BaseLearner
from ding.worker.collector import base_serial_collector, base_serial_evaluator

model_save_path = "trained_models"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initial_setup():
    os.makedirs(model_save_path, exist_ok=True)
    logging.info(f"Using device: {device}")

def generate_model_name(params: Dict[str, any]) -> str:
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"model_{param_hash}.zip"

def train_model(train_data, timesteps=50000):
    # Create DI-engine environment
    env = create_di_env(df=train_data, window_size=10, frame_bound=(10, len(train_data)))
    env = DingEnvWrapper(env)
    
    # Get and update config
    cfg = get_default_config()
    cfg.policy.model.obs_shape = env.observation_space.shape[0]
    cfg.policy.model.action_shape = env.action_space.n

    # Create policy and training components
    policy = create_forex_agent(env, cfg)
    learner = BaseLearner(cfg.policy, policy.learn_mode, env.env_id)
    collector = base_serial_collector(cfg.collecting, env.env_id, policy.collect_mode)
    evaluator = base_serial_evaluator(cfg.policy, env.env_id, policy.eval_mode)
    
    # Training loop
    while not evaluator.should_stop():
        new_data = collector.collect()
        learner.train(new_data)
        if learner.train_iter % 1000 == 0:  # Save checkpoint periodically
            save_path = os.path.join(model_save_path, f"checkpoint_{learner.train_iter}.pth")
            policy.save(save_path)
            
    return policy
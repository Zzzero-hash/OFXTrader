from typing import Optional, Dict
from ding.model.common import RegressionHead, DiscreteHead, DuelingHead
from ding.torch_utils import MLP
from ding.config import compile_config, read_config
from ding.policy import SACPolicy
import torch
import torch.nn as nn
import logging

class ForexTradingNetwork(nn.Module):
    def __init__(
            self,
            obs_shape: int,
            action_shape: int,
            hidden_size_list: list = [128, 128],
            activation: Optional[torch.nn.Module] = torch.nn.ReLU(),
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        # Actor network with discrete output
        self.actor = nn.Sequential(
            MLP(
                in_channels=obs_shape,
                hidden_channels=hidden_size_list,
                out_channels=64,
                activation=activation,
            ),
            DiscreteHead(64, action_shape)
        )
        
        # Twin Q networks for TD3/SAC
        critic_input_size = obs_shape + action_shape
        self.critic = nn.ModuleList([
            nn.Sequential(
                MLP(
                    in_channels=critic_input_size,
                    hidden_channels=hidden_size_list,
                    out_channels=64,
                    activation=activation,
                ),
                RegressionHead(64, 1)
            ) for _ in range(2)
        ])

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        action = self.actor(obs)
        obs_action = torch.cat([obs, action], dim=1)
        q_value_1 = self.critic[0](obs_action)
        q_value_2 = self.critic[1](obs_action)
        return {
            'action': action,
            'q_value_1': q_value_1,
            'q_value_2': q_value_2
        }

def create_forex_agent(env, cfg):
    # Create model config
    model = ForexTradingNetwork(
        obs_shape=env.observation_space.shape[0],
        action_shape=env.action_space.n,
    )
    
    # Use SAC policy
    policy = SACPolicy(cfg.policy, model=model)
    return policy

def get_default_config():
    cfg = {
        'policy': {
            'type': 'sac',  # or 'td3'
            'model': {
                'obs_shape': None,
                'action_shape': None,
            },
            'default_config': {},
            'learn': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'target_update_interval': 1,
                'target_update_tau': 0.005,
                'update_per_collect': 1,
                'auto_alpha': True,
            },
            'collect': {
                'n_sample': 64,
                'unroll_len': 1,
            },
            'eval': {
                'evaluator': {
                    'eval_freq': 1000,
                },
            },
            'other': {
                'replay_buffer': {
                    'replay_buffer_size': 100000,
                },
            },
        },
    }
    try:
        cfg['policy']['default_config'] = cfg['policy']
        return compile_config(cfg)
    except Exception as e:
        logging.error("Config error: {}".format(e))
        raise e

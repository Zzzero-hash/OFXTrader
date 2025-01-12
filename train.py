import os
import logging
from typing import List, Dict
import json
import hashlib
import optuna
import subprocess
import sys
import pandas as pd
import torch
import gymnasium as gym
import gym_anytrading
from stable_baselines3 import PPO
from forex_env import create_env
from finrl.agents.stablebaselines3.models import DRLAgent
from fetch_data import fetch_oanda_candles_range, add_technical_indicators
from evaluation_utils import test_model
from stable_baselines3.common.callbacks import BaseCallback

# Define global variables
model_save_path = "trained_models"
device = 'cpu'

logging.basicConfig(level=logging.DEBUG)

def initial_setup():
    global device, model_save_path
    # Enforced CUDA usage
    if not torch.cuda.is_available():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "stable-baselines3[cuda]"])
        except subprocess.CalledProcessError as e:
            raise EnvironmentError("CUDA is not available and failed to install stable-baselines3 with CUDA support.") from e

    # Verify CUDA availability using nvidia-smi
    try:
        cuda_available = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).returncode == 0
    except FileNotFoundError:
        cuda_available = False

    # Add dynamic device assignment
    device = 'cuda' if cuda_available else 'cpu'
    logging.info(f"Using device: {device}")

    # Create a directory to save the trained model
    model_save_path = "trained_models"
    os.makedirs(model_save_path, exist_ok=True)

def list_available_models() -> List[str]:
    """List all available pre-trained models."""
    models = [f for f in os.listdir(model_save_path) if f.endswith('.zip')]
    return models

def generate_model_name(params: Dict[str, any]) -> str:
    """Generate a unique model name based on training parameters."""
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"model_{param_hash}.zip"

def load_model(model_name: str) -> PPO:
    """Load a pre-trained model."""
    model_file = os.path.join(model_save_path, model_name)
    if os.path.exists(model_file):
        model = PPO.load(model_file, device=device)
        logging.info(f"Loaded model from {model_file}")
        return model
    else:
        raise FileNotFoundError(f"Model file {model_file} not found.")

class EnvLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EnvLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            current_reward = self.training_env.get_attr("current_step_reward")[0] \
                if hasattr(self.training_env.unwrapped, "current_step_reward") else None
            logging.debug(f"[Callback] Step: {self.n_calls} Reward: {current_reward}")
            print(f"[Callback] Step: {self.n_calls}, Reward: {current_reward}")  # Immediate stdout feedback
        return True

def train_model(train_data, use_optuna=True, n_trials=10, timesteps=50000, stop_loss_percent=0.02):
    params = {
        "timesteps": timesteps,
        "stop_loss_percent": stop_loss_percent,
        "use_optuna": use_optuna,
        "n_trials": n_trials
    }
    if use_optuna:
        model = hyperparam_tuning(train_data, n_trials=n_trials, stop_loss_percent=stop_loss_percent)
    else:
        model = PPO(
            "MlpPolicy",
            create_env(df=train_data,
                       window_size=10,
                       frame_bound=(10, len(train_data)),
                       unit_side='right'),
            verbose=1,
            device=device
        )
        # Pass our custom callback
        model.learn(total_timesteps=timesteps, callback=EnvLoggingCallback())

    if device == 'cuda':
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')

    # Save the trained model
    model_name = generate_model_name(params)
    model_file = os.path.join(model_save_path, model_name)
    try:
        model.save(model_file)
        logging.info(f"Model saved to {model_file}")
    except Exception as e:
        logging.error(f"Failed to save model to {model_file}: {e}")

    return model_name

def hyperparam_tuning(train_data: pd.DataFrame, n_trials: int = 10, stop_loss_percent: float = 0.02) -> PPO:
    """
    Perform hyperparameter tuning using Optuna.

    Parameters:
    train_data (pd.DataFrame): The training data containing historical price and technical indicators.
    n_trials (int): The number of trials for Optuna hyperparameter tuning. Default is 10.

    Returns:
    PPO: The trained PPO model with the best hyperparameters.
    """
    def objective(trial):
        logging.info("Starting a new trial...")
        # Narrow down parameter ranges
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
        n_steps = trial.suggest_int('n_steps', 256, 512)
        gamma = trial.suggest_float('gamma', 0.98, 0.995)
        ent_coef = trial.suggest_float('ent_coef', 1e-5, 1e-4)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)

        # Additional PPO parameters
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 0.7)
        vf_coef = trial.suggest_float('vf_coef', 0.4, 0.6)

        env_train = create_env(df=train_data,
                               window_size=10,
                               frame_bound=(10, len(train_data)),
                               unit_side='right')

        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            batch_size=batch_size,
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            policy_kwargs={"net_arch": [128, 128]},  # Deeper network
            device=device
        )

        model.learn(total_timesteps=20000)
        train_reward, _ = test_model(train_data, model, n_eval_episodes=5)
        logging.info(f"Trial completed with reward: {train_reward}")
        return train_reward

    study = optuna.create_study(direction="maximize")
    logging.info("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=n_trials, timeout=3600)  # Set a timeout of 1 hour
    best_params = study.best_params
    logging.info("Best hyperparameters: %s", best_params)

    # Retrain final model with best parameters
    final_env = create_env(df=train_data,
                           window_size=10,
                           frame_bound=(10, len(train_data)),
                           unit_side='right')
    final_model = PPO(
        "MlpPolicy",
        final_env,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        ent_coef=best_params['ent_coef'],
        batch_size=best_params['batch_size'],
        clip_range=best_params['clip_range'],
        max_grad_norm=best_params['max_grad_norm'],
        vf_coef=best_params['vf_coef'],
        policy_kwargs={"net_arch": [128, 128]},  # Deeper network
        device=device
    )
    final_model.learn(total_timesteps=100000)

    # Save the final model with best parameters
    model_name = generate_model_name(best_params)
    model_file = os.path.join(model_save_path, model_name)
    try:
        final_model.save(model_file)
        logging.info(f"Model with best hyperparameters saved to {model_file}")
    except Exception as e:
        logging.error(f"Failed to save model with best hyperparameters to {model_file}: {e}")

    return final_model
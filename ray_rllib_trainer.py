import os
import ray
import torch
import numpy as np
import logging
from ray import tune
from ray.air import session
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.optuna import OptunaSearch
from gymnasium.utils.env_checker import check_env
from forex_env import ForexEnv
import gymnasium as gym

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shared Configuration
INSTRUMENTS = ['EUR_USD', 'USD_JPY']
ENV_BASE_CONFIG = {
    "instruments": INSTRUMENTS,
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "granularity": "M1",
    "initial_balance": 1000,
    "leverage": 50,
    "spread_pips": 0.001,
    "render_frequency": 10000
}

NUM_ROLLOUT_WORKERS = 2  # Number of parallel workers

TUNING_SETTING = {
    "n_trials": 5,      # Number of tuning trials
    "tune_epochs": 10,  # Epochs for tuning 
    "final_epochs": 20, # Epochs for final training
    "resource": {"num_gpus": 1 if torch.cuda.is_available() else 0}  # Dynamic GPU usage
}

# Environment Registration
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# Training Function
def train_model(config):
    """Train the PPO model for multi-instrument Forex trading."""
    try:
        logger.info(f"Training started - CUDA: {torch.cuda.is_available()}")

        # Create a clean environment config with only the parameters for ForexEnv
        env_config = {
            "instruments": INSTRUMENTS,
            "data_array": ENV_BASE_CONFIG["data_array"],
            "feature_names": ENV_BASE_CONFIG["feature_names"],
            "initial_balance": ENV_BASE_CONFIG["initial_balance"],
            "leverage": ENV_BASE_CONFIG["leverage"],
            "spread_pips": ENV_BASE_CONFIG["spread_pips"],
            "render_frequency": ENV_BASE_CONFIG["render_frequency"]
        }
        # Configure PPO for multi-instrument setup
        algo_config = PPOConfig()
        algo_config.environment(env="forex-v0", env_config=env_config)
        algo_config.framework("torch")
        
        # Resource allocation
        algo_config.resources(
            num_gpus=1 if torch.cuda.is_available() else 0,  
            num_cpus_per_worker=1,
            num_gpus_per_worker=0 
        )
        
        # Training settings
        algo_config.training(
            train_batch_size=config["train_batch_size"],
            lr=config["lr"],
            gamma=config["gamma"],
            model={
                "custom_model": None,
                "use_lstm": config["model_use_lstm"],
                "lstm_cell_size": config["model"]["lstm_cell_size"],
                "fcnet_hiddens": config["model"]["fcnet_hiddens"],
                "fcnet_activation": config["model"]["fcnet_activation"],
                "vf_share_layers": True
            }
        )
        
        algo_config.rollouts(
            num_rollout_workers=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length='auto'
        )
        
        algo_config.exploration(
            explore=True,
            exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.02}
        )

        # Build and train the model
        trainer = algo_config.build()
        best_mean_reward = -float('inf')

        for epoch in range(config["num_epochs"]):
            result = trainer.train()
            mean_reward = result.get("episode_reward_mean", -float('inf'))
            logger.info(f"Epoch {epoch + 1}: Mean Reward = {mean_reward}")
            if not np.isfinite(mean_reward):
                logger.warning(f"Non-finite reward: {mean_reward}")
                mean_reward = -float('inf')
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Mean Reward = {mean_reward}, "
                        f"Episodes This Iter = {result.get('episodes_this_iter', 0)}")

            if mean_reward > best_mean_reward and np.isfinite(mean_reward):
                best_mean_reward = mean_reward
            
            # Just report metrics every iteration without checkpoints
            session.report(mean_reward=mean_reward)
        
        # Final report with final mean reward
        session.report(mean_reward=best_mean_reward)
        
        trainer.stop()  # Clean up resources

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# Main Execution
def train_forex_model():
    """Execute the training pipeline with hyperparameter tuning and final training."""
    try:
        # Create a clean environment config with only the parameters for ForexEnv
        env_config = {
            "instruments": INSTRUMENTS,
            "data_array": ENV_BASE_CONFIG["data_array"],
            "feature_names": ENV_BASE_CONFIG["feature_names"],
            "initial_balance": ENV_BASE_CONFIG["initial_balance"],
            "leverage": ENV_BASE_CONFIG["leverage"],
            "spread_pips": ENV_BASE_CONFIG["spread_pips"],
            "render_frequency": ENV_BASE_CONFIG["render_frequency"]
        }

        # Base configuration for final training
        base_config = {
            "env": "forex-v0",
            "env_config": env_config,
            "num_epochs": TUNING_SETTING["tune_epochs"],
            "model_use_lstm": True,
            "num_rollout_workers": NUM_ROLLOUT_WORKERS
        }

        # Hyperparameter search space 
        search_space = {
            "train_batch_size": tune.choice([2048, 4096]),
            "lr": tune.loguniform(1e-5, 1e-3),
            "gamma": tune.uniform(0.9, 0.999),
            "model": {
                "custom_model": None,
                "use_lstm": True,
                "lstm_cell_size": tune.choice([128, 256]),
                "fcnet_hiddens": tune.choice([(128, 128), (256, 256)]),
                "fcnet_activation": tune.choice(["relu", "tanh"])
            }
        }
        
        # Optuna search for hyperparameter tuning
        optuna_search = OptunaSearch(metric="mean_reward", mode="max")
        
        # Resource allocation with placement groups
        # Each trial will use 1 CPU and 1 GPU, plus 1 CPU for each rollout worker
        pg = tune.PlacementGroupFactory(
            [{"CPU": 1, "GPU": 0.5}] + [{"CPU": 1}] * NUM_ROLLOUT_WORKERS
        )
        
        # Hyperparameter tuning
        analysis = tune.run(
            train_model,
            config={**base_config, **search_space},
            search_alg=optuna_search,
            num_samples=TUNING_SETTING["n_trials"],
            resources_per_trial=pg,  # Use the placement group factory
            local_dir=os.path.abspath("logs"),
            verbose=1,
            max_failures=3
        )

        # Get best configuration
        best_trial = analysis.get_best_trial("mean_reward", mode="max")
        best_config = best_trial.config
        logger.info(f"Best trial - Value: {best_trial.last_result['mean_reward']}, Params: {best_config}")

        # Update base config with best parameters for final training
        best_config["num_epochs"] = TUNING_SETTING["final_epochs"]

        # Final training with best config
        final_analysis = tune.run(
            train_model,
            config=best_config,
            resources_per_trial=pg,  
            local_dir=os.path.abspath("logs"),
            stop={"training_iteration": TUNING_SETTING["final_epochs"]},
            verbose=1,
            name="forex_final_training",
            max_failures=3
        )
        
        # Log final results
        best_final_trial = final_analysis.get_best_trial("mean_reward", mode="max")
        logger.info(f"Best final trial - Value: {best_final_trial.last_result['mean_reward']}")

        # Save the final model
        final_config = PPOConfig()
        final_config.environment(env="forex-v0", env_config=env_config)
        final_config.framework("torch")
        final_config.resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus_per_worker=1
        )
        final_config.training(
            train_batch_size=best_config["train_batch_size"],
            lr=best_config["lr"],
            gamma=best_config["gamma"],
            model=best_config["model"]
        )
        final_trainer = final_config.build()
        
        for i in range(10):  # Short final training
            final_trainer.train()
        
        # Save the model
        save_dir = os.path.join("trained_forex_model", "final_model")
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        save_path = final_trainer.save(save_dir)
        logger.info(f"Final model saved at: {save_path}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:        # Initialize Ray and fetch data for training
        from data_handler import DataHandler 
        from ray.tune import register_env 
        data_handler = DataHandler()
        register_env("forex-v0", lambda config: ForexEnv(**config))
        data_array, feature_names = data_handler.get_data(
            instruments=INSTRUMENTS,
            start_date=ENV_BASE_CONFIG["start_date"],
            end_date=ENV_BASE_CONFIG["end_date"],
            granularity=ENV_BASE_CONFIG["granularity"],
            window_size=14
        )
        ENV_BASE_CONFIG["data_array"] = data_array
        ENV_BASE_CONFIG["feature_names"] = feature_names
        print(f"Feature names: {feature_names}")
        print(f"Data array shape: {data_array.shape}")

        ray.init(
            num_cpus=4,
            num_gpus=TUNING_SETTING["resource"]["num_gpus"],
            ignore_reinit_error=True
        )
        train_forex_model()
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
    finally:
        ray.shutdown()
        logger.info("Ray resources released")
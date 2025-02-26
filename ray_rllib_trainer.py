import os
import ray
import torch
import numpy as np
import logging
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.optuna import OptunaSearch
from gymnasium.utils.env_checker import check_env
from forex_env import ForexEnv
import optuna
import gymnasium as gym
import tempfile

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shared Configuration
ENV_BASE_CONFIG = {
    "instrument": "EUR_USD",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "granularity": "M1",
    "window_size": 14,  # Sequence length for LSTM
    "initial_balance": 1000,
    "leverage": 50  
}

TUNING_SETTING = {
    "n_trials": 10,
    "tune_epochs": 20,  # Epochs for hyperparameter tuning
    "final_epochs": 100,  # Epochs for final training
    "resource": {"num_gpus": 1 if torch.cuda.is_available() else 0}  # Dynamic GPU usage
}

# Environment Registration
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# Training Function
def train_model(config):
    """Train the PPO model and report results."""
    try:
        logger.info(f"Training started - CUDA: {torch.cuda.is_available()}")
        
        # Configure PPO with RLlib's best practices
        algo_config = PPOConfig()
        algo_config.environment(env="forex-v0", env_config=ENV_BASE_CONFIG)
        algo_config.framework("torch")
        algo_config.resources(**TUNING_SETTING["resource"])
        algo_config.training(
            train_batch_size=config["train_batch_size"],
            lr=config["lr"],
            gamma=config["gamma"],
            model=config["model"]
        )
        algo_config.rollouts(
            num_rollout_workers=config.get("num_rollout_workers", 4),
            rollout_fragment_length='auto'  # Match max_steps for episode completion
        )
        algo_config.exploration(
            explore=True,
            exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.02}
        )

        trainer = algo_config.build()
        best_mean_reward = -float('inf')

        for epoch in range(config["num_epochs"]):
            result = trainer.train()
            mean_reward = result.get("episode_reward_mean", -float('inf'))
            if not np.isfinite(mean_reward):
                logger.warning(f"Non-finite reward: {mean_reward}")
                mean_reward = -float('inf')
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Mean Reward = {mean_reward}, "
                        f"Episodes This Iter = {result.get('episodes_this_iter', 0)}")

            if mean_reward > best_mean_reward and np.isfinite(mean_reward):
                best_mean_reward = mean_reward
            
            # Create checkpoint for intermediate reporting
            if epoch % 5 == 0:
                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_path = trainer.save(temp_dir)
                    tune.report(mean_reward=mean_reward, checkpoint=checkpoint_path)
        
        # Final checkpoint and report
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = trainer.save(temp_dir)
            tune.report(mean_reward=best_mean_reward, checkpoint=checkpoint_path)
        
        trainer.stop()  # Clean up resources

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# Main Execution
def train_forex_model():
    """Execute the training pipeline with Optuna tuning and final training."""
    try:
        # Base configuration for final training
        base_config = {
            "env": "forex-v0",
            "env_config": ENV_BASE_CONFIG,
            "num_epochs": TUNING_SETTING["tune_epochs"],
            "model_use_lstm": True,
            "model_max_seq_len": ENV_BASE_CONFIG["window_size"],
            "num_rollout_workers": 3
        }

        # Define the search space for Optuna
        search_space = {
            "train_batch_size": tune.choice([2048, 4096, 8192]),
            "lr": tune.loguniform(1e-6, 1e-3),
            "gamma": tune.uniform(0.9, 0.999),
            "model": {
                "custom_model": None,
                "use_lstm": True,
                "lstm_cell_size": tune.choice([128, 256, 512, 1024, 2048]),
                "max_seq_len": ENV_BASE_CONFIG["window_size"],
                "fcnet_hiddens": tune.choice([[256, 256], [512, 512], [1024, 512], [1024, 1024]]),
                "fcnet_activation": tune.choice(["relu", "tanh", "swish"])
            }
        }
        
        # Setup hyperparameter search with Optuna - using minimal parameters
        optuna_search = OptunaSearch(metric="mean_reward", mode="max")
        
        # Simple tune.run with most basic parameters for compatibility
        analysis = tune.run(
            train_model,
            config={**base_config, **search_space},
            search_alg=optuna_search,
            num_samples=TUNING_SETTING["n_trials"],
            resources_per_trial={
                "cpu": 1, 
                "gpu": TUNING_SETTING["resource"]["num_gpus"] / (base_config["num_rollout_workers"] + 1)
            },
            local_dir=os.path.abspath("logs"),
            verbose=1,
            checkpoint_freq=5,
            keep_checkpoints_num=1,
            checkpoint_score_attr="mean_reward",
            max_failures=3
        )

        # Get best configuration
        best_trial = analysis.get_best_trial("mean_reward", mode="max")
        best_config = best_trial.config
        logger.info(f"Best trial - Value: {best_trial.last_result['mean_reward']}, Params: {best_config}")

        # Update base config with best parameters for final training
        best_config["num_epochs"] = TUNING_SETTING["final_epochs"]

        # Simplified final training with best config
        final_analysis = tune.run(
            train_model,
            config=best_config,
            local_dir=os.path.abspath("logs"),
            resources_per_trial={
                "cpu": 1, 
                "gpu": TUNING_SETTING["resource"]["num_gpus"]
            },
            checkpoint_freq=10,
            keep_checkpoints_num=1,
            checkpoint_score_attr="mean_reward",
            stop={"training_iteration": TUNING_SETTING["final_epochs"]},
            verbose=1,
            name="forex_final_training",
            max_failures=3
        )
        
        # Get the best checkpoint
        best_final_trial = final_analysis.get_best_trial("mean_reward", mode="max")
        best_checkpoint = final_analysis.get_best_checkpoint(best_final_trial, "mean_reward", "max")
        logger.info(f"Best checkpoint: {best_checkpoint}")

        # Save final model
        final_trainer = PPOConfig().update_from_dict(best_config).build()
        final_trainer.restore(best_checkpoint)
        save_path = final_trainer.save(os.path.join("trained_forex_model", "final_model"))
        logger.info(f"Final model saved at: {save_path}")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        env = gym.make("forex-v0", **ENV_BASE_CONFIG)
        check_env(env.unwrapped)
        logger.info("Environment validated successfully")
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
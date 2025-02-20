import os
import numpy as np
import ray
import torch
import logging
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.optuna import OptunaSearch
from gymnasium.utils.env_checker import check_env
from forex_env import ForexEnv
import optuna
import gymnasium as gym
from ray.tune import PlacementGroupFactory
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import Tuner

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
    "leverage": 10  # Reduced from 50 for stability
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
            num_rollout_workers=4,
            rollout_fragment_length=500  # Match max_steps for episode completion
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

        tune.report(mean_reward=best_mean_reward)
        trainer.stop()  # Clean up resources
        return {"mean_reward": best_mean_reward}

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
            "model": {
                "use_lstm": True,
                "max_seq_len": ENV_BASE_CONFIG["window_size"],
                "lstm_cell_size": 128,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "lr": 0.00005,
            "gamma": 0.99,
            "train_batch_size": 4000
        }

        # Define the search space for Optuna
        optuna_search = OptunaSearch(
            space={
                "model": {
                    "use_lstm": True,
                    "max_seq_len": ENV_BASE_CONFIG["window_size"],
                    "lstm_cell_size": tune.choice([256, 512]),
                    "fcnet_hiddens": tune.choice([[256, 256], [512, 256], [512, 512], [1024, 512]]),
                    "fcnet_activation": tune.choice(["relu", "tanh"])
                },
                "lr": tune.loguniform(1e-5, 1e-3),
                "gamma": tune.uniform(0.9, 0.999),
                "train_batch_size": tune.choice([2048, 4096, 8192])
            },
            metric="mean_reward",
            mode="max",
            storage="sqlite:///forex_study.db",  # Use the RDB storage
            study_name="forex_ppo",
            load_if_exists=True
        )

        # Configure the Tuner
        tuner = tune.Tuner(
            tune.with_parameters(train_model),  # Pass the training function directly
            run_config=ray.tune.RunConfig(
                name="forex_ppo",
                local_dir=os.path.abspath("logs"),
                stop={"training_iteration": TUNING_SETTING["tune_epochs"]},
                callbacks=[ray.tune.logger.TBXLoggerCallback()],
            ),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=TUNING_SETTING["n_trials"],
            ),
            param_space=base_config  # Pass the base config as the param_space
        )
        results = tuner.fit()

        # Get best configuration
        best_result = results.get_best_trial("mean_reward", mode="max")
        best_config = best_result.config
        logger.info(f"Best trial - Value: {best_result.metrics['mean_reward']}, Params: {best_config}")

        # Update base config with best parameters for final training
        best_config["num_epochs"] = TUNING_SETTING["final_epochs"]

        # Resource allocation for final training
        num_rollout_workers = 1
        resources_per_trial = PlacementGroupFactory(
            [{"CPU": 2.0, "GPU": 0.5 if TUNING_SETTING["resource"]["num_gpus"] > 0 else 0.0}] +
            [{"CPU": 1.0, "GPU": 0.0}] * num_rollout_workers
        )

        # Final training with best config
        analysis = tune.run(
            train_model,
            config=best_config,
            local_dir=os.path.abspath("logs"),
            callbacks=DEFAULT_LOGGERS,
            stop={"training_iteration": TUNING_SETTING["final_epochs"]},
            checkpoint_at_end=True,
            checkpoint_freq=10,
            keep_checkpoints_num=1,
            checkpoint_score_attr="mean_reward",
            metric="mean_reward",
            mode="max",
            verbose=1,
            resources_per_trial=resources_per_trial
        )

        best_trial = analysis.get_best_trial("mean_reward", mode="max")
        best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="mean_reward", mode="max")
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
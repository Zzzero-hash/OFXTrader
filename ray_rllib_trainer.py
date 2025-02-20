import os
import ray
import torch
import logging
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.utils.env_checker import check_env
from forex_env import ForexEnv
import optuna
import gymnasium as gym
from ray.tune import PlacementGroupFactory
from ray.air import session

# Configure logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================
# Shared Configuration
# ======================
ENV_BASE_CONFIG = {
    "instrument": "EUR_USD",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "granularity": "M1",
    "window_size": 14,  # Sequence length for LSTM
    "initial_balance": 1000,
    "leverage": 50
}

TUNING_SETTINGS = {
    "n_trials": 10,
    "tune_epochs": 20,  # Epochs for hyperparameter tuning
    "final_epochs": 100,  # Epochs for final training
    "resource": {"num_gpus": 1 if torch.cuda.is_available() else 0}  # Dynamic GPU usage
}

# ======================
# Environment Registration
# ======================
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# ======================
# Training Function
# ======================
def train_model(config, tune_mode=True):
    """Train the PPO model and report results."""
    try:
        logger.info(f"Training started - Tune mode: {tune_mode}, CUDA: {torch.cuda.is_available()}")

        # Configure PPO with RLlib's best practices (ref: RLlib docs)
        algo_config = PPOConfig()
        algo_config.environment(env="forex-v0", env_config=ENV_BASE_CONFIG)
        algo_config.framework("torch")
        algo_config.resources(**TUNING_SETTINGS["resource"])
        algo_config.training(
            train_batch_size=config["train_batch_size"],
            lr=config["lr"],
            gamma=config["gamma"],
            model=config["model"]
        )
        algo_config.rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=200  # Recommended default from RLlib docs
        )
        algo_config.exploration(explore=True)

        trainer = algo_config.build()
        best_mean_reward = -float('inf')

        for epoch in range(config["num_epochs"]):
            result = trainer.train()
            mean_reward = result.get("episode_reward_mean", -float('inf'))
            logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}: Mean Reward = {mean_reward}, "
                       f"Episodes This Iter = {result.get('episodes_this_iter', 0)}")

            if mean_reward > best_mean_reward and mean_reward != -float('inf'):
                best_mean_reward = mean_reward

        if tune_mode:
            session.report({"episode_reward_mean": best_mean_reward})
            logger.info(f"Tune mode - Reported episode_reward_mean: {best_mean_reward}")
        else:
            logger.info(f"Final mode - Best mean reward: {best_mean_reward}")
            return trainer

        trainer.stop()  # Clean up resources
        return best_mean_reward

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

# ======================
# Optuna Optimization
# ======================
def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    try:
        hyperparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "fcnet_hiddens": trial.suggest_categorical("fcnet_hiddens", ["256,256", "512,256", "512,512"]),
            "train_batch_size": trial.suggest_categorical("train_batch_size", [2048, 4096, 8192]),
            "num_epochs": TUNING_SETTINGS["tune_epochs"],
            "model": {
                "fcnet_hiddens": None,  # Set below
                "fcnet_activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "use_lstm": True,
                "max_seq_len": ENV_BASE_CONFIG["window_size"],
                "lstm_cell_size": trial.suggest_categorical("lstm_cell_size", [256, 512])  # Expanded range
            }
        }

        hyperparams["model"]["fcnet_hiddens"] = [int(x) for x in hyperparams["fcnet_hiddens"].split(",")]
        logger.info(f"Trial {trial.number}: Hyperparameters = {hyperparams}")

        result = train_model(hyperparams, tune_mode=True)
        print(result)
        return result

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

# ======================
# Main Execution
# ======================
def train_forex_model():
    """Execute the training pipeline with Optuna tuning and final training."""
    try:
        # Base configuration for final training
        base_config = {
            "env": "forex-v0",
            "env_config": ENV_BASE_CONFIG,
            "num_epochs": TUNING_SETTINGS["tune_epochs"],
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

        # Optuna optimization with persistent storage (ref: Optuna docs)
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///forex_study.db",
            study_name="forex_ppo",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=TUNING_SETTINGS["n_trials"], n_jobs=1)  # Sequential for stability

        logger.info(f"Best trial - Value: {study.best_value}, Params: {study.best_params}")

        # Update config with best hyperparameters
        best_config = base_config.copy()
        best_config.update({
            "lr": study.best_params["lr"],
            "gamma": study.best_params["gamma"],
            "train_batch_size": study.best_params["train_batch_size"],
            "model": {
                **best_config["model"],
                "fcnet_hiddens": [int(x) for x in study.best_params["fcnet_hiddens"].split(",")],
                "fcnet_activation": study.best_params["activation"],
                "lstm_cell_size": study.best_params["lstm_cell_size"]
            },
            "num_epochs": TUNING_SETTINGS["final_epochs"]
        })

        # Resource allocation with RLlib recommendations
        num_rollout_workers = 1
        resources_per_trial = PlacementGroupFactory(
            [{"CPU": 2.0, "GPU": 0.5 if TUNING_SETTINGS["resource"]["num_gpus"] > 0 else 0.0}] +
            [{"CPU": 1.0, "GPU": 0.0}] * num_rollout_workers
        )

        # Final training with Tune
        analysis = tune.run(
            tune.with_parameters(train_model, tune_mode=False),
            config=best_config,
            local_dir=os.path.abspath("logs"),
            stop={"training_iteration": TUNING_SETTINGS["final_epochs"]},
            checkpoint_at_end=True,
            checkpoint_freq=10,  # Regular checkpoints for recovery
            keep_checkpoints_num=1,
            checkpoint_score_attr="episode_reward_mean",
            metric="episode_reward_mean",
            mode="max",
            verbose=1,
            resources_per_trial=resources_per_trial
        )

        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="episode_reward_mean", mode="max")
        logger.info(f"Best checkpoint: {best_checkpoint}")

        # Save final model
        final_trainer = train_model(best_config, tune_mode=False)
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
            num_gpus=TUNING_SETTINGS["resource"]["num_gpus"],
            ignore_reinit_error=True  # Avoid errors on re-run
        )
        train_forex_model()

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
    finally:
        ray.shutdown()
        logger.info("Ray resources released")
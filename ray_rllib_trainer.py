import os
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.utils.env_checker import check_env
from ray.tune.logger import TBXLoggerCallback
from forex_env import ForexEnv
import optuna
import gymnasium as gym
from ray.tune import PlacementGroupFactory

# ======================
# Shared Configuration
# ======================
ENV_BASE_CONFIG = {
    "instrument": "EUR_USD",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "granularity": "M1",
    "window_size": 14,
    "initial_balance": 10000,
    "leverage": 50
}

TUNING_SETTINGS = {
    "n_trials": 10,
    "tune_epochs": 20,  # Increased to allow episodes to complete
    "final_epochs": 100,
    "resource": {"num_gpus": 1} # Specify GPU usage here
}

# ======================
# Environment Registration
# ======================
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# ======================
# Training Function (Modified for new API stack)
# ======================
def train_model(config, tune_mode=True, render_during_train=True):
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Configure the PPO algorithm using the new API stack
    algo_config = (
        PPOConfig()
        .environment(env="forex-v0", env_config=ENV_BASE_CONFIG)
        .framework("torch")
        .resources(**TUNING_SETTINGS["resource"])
        .training(
            train_batch_size=config["train_batch_size"],
            lr=config["lr"],
            gamma=config["gamma"],
        )
        .rollouts(num_rollout_workers=1)  # Add rollout workers
        .exploration(explore=True)  # Enable exploration
        .debugging(
            logger_config={
                "wandb": {
                    "project": "forex_rllib",
                    "api_key_file": "~/.wandb_api_key",
                    "log_config": True,
                }
            }
        )
    )

    # Build the trainer
    trainer = algo_config.build()

    # Track the best reward across all training iterations
    best_mean_reward = -float('inf')

    for epoch in range(config["num_epochs"]):
        result = trainer.train()

        # Safely extract reward with comprehensive fallbacks
        mean_reward = result.get("episode_reward_mean", result.get("episode_reward", -float('inf'))) or -float('inf')

        # Update best reward
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

    # Report the best reward seen during tuning
    if tune_mode:
        tune.report(mean_reward=best_mean_reward)

    return trainer

# ======================
# Optuna Optimization (Fixed)
# ======================
def objective(trial):
    hyperparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "fcnet_hiddens": trial.suggest_categorical(
            "fcnet_hiddens", ["256,256", "512,256", "512,512"]
        ),
        "train_batch_size": trial.suggest_categorical(
            "train_batch_size", [2048, 4096, 8192]
        ),
        "num_epochs": TUNING_SETTINGS["tune_epochs"],
        "model": {  # Model config using new API
            "fcnet_hiddens": None,  # To be set below
            "fcnet_activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]
            )
        }
    }

    # Use the chosen fcnet_hiddens value without removing it so that it is saved in best_params
    fcnet_str = hyperparams["fcnet_hiddens"]
    hyperparams["model"]["fcnet_hiddens"] = [
        int(x) for x in fcnet_str.split(",")
    ]

    # Train and automatically report via tune.report()
    trainer = train_model(hyperparams, tune_mode=True)

# ======================
# Main Execution
# ======================
def train_forex_model():
    config = PPOConfig().to_dict()
    config["logger_config"] = {"logdir": "logs"}
    config["env"] = "forex-v0"
    config["env_config"] = ENV_BASE_CONFIG
    config["num_epochs"] = TUNING_SETTINGS["tune_epochs"]

    # Define resources per trial
    num_rollout_workers = 1  # Use the same number as in PPOConfig
    resources_per_trial = PlacementGroupFactory(
        [{'CPU': 1.0, 'GPU': 0.0 if TUNING_SETTINGS["resource"]["num_gpus"] == 0 else 0.5}] +  # Driver resources
        [{'CPU': 1.0, 'GPU': 0.0 if TUNING_SETTINGS["resource"]["num_gpus"] == 0 else 0.5}] * num_rollout_workers   # Worker resources
    )

    analysis = tune.run(
        train_model,
        config=config,
        storage_path=f"file://{os.path.abspath('logs')}",
        stop={"training_iteration": 100},
        checkpoint_at_end=False,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
        verbose=1,
        resources_per_trial=resources_per_trial  # Add the resources_per_trial argument
    )

    best_checkpoint = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
        mode="max"
    )

    print(f"Best checkpoint saved at: {best_checkpoint}")

    # Save the best performing model
    final_trainer = train_model(analysis.best_config, tune_mode=False)
    final_trainer.restore(best_checkpoint)
    final_trainer.save("trained_forex_model")

    # Log final training results
    print("Training completed. Best model saved.")

if __name__ == "__main__":
    env = gym.make('forex-v0', **ENV_BASE_CONFIG)
    check_env(env.unwrapped)

    ray.init(num_cpus=4, num_gpus=1)
    train_forex_model()
    ray.shutdown()
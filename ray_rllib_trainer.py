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
from ray.air import session

# ======================
# Shared Configuration
# ======================
ENV_BASE_CONFIG = {
    "instrument": "EUR_USD",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "granularity": "M1",
    "window_size": 14,  # Desired sequence length for the LSTM
    "initial_balance": 1000,
    "leverage": 50
}

TUNING_SETTINGS = {
    "n_trials": 10,
    "tune_epochs": 20,  # Increased to allow episodes to complete
    "final_epochs": 100,
    "resource": {"num_gpus": 1}  # Specify GPU usage here
}

# ======================
# Environment Registration
# ======================
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# ======================
# Training Function (Using Native Preprocessing and Built-in LSTM)
# ======================
def train_model(config, tune_mode=True, render_during_train=True):
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Configure the PPO algorithm using the new API stack.
    # Here we allow RLlib's native preprocessor to operate (i.e. we do not disable it)
    # and we enable the built-in recurrent (LSTM) wrapper by setting "use_lstm": True,
    # "max_seq_len" to ENV_BASE_CONFIG["window_size"] (14), and "lstm_cell_size" to 128.
    algo_config = PPOConfig()
    algo_config.environment(env="forex-v0", env_config=ENV_BASE_CONFIG)
    algo_config.framework("torch") 
    algo_config.resources(**TUNING_SETTINGS["resource"]) 
    algo_config.training(
        train_batch_size=config["train_batch_size"],
        lr=config["lr"],
        gamma=config["gamma"],
    )
    algo_config.rollouts(num_rollout_workers=1)  
    algo_config.exploration(explore=True)  
    algo_config.model.update(config["model"])

    # Build the trainer
    trainer = algo_config.build()

    # Track the best reward across training iterations.
    best_mean_reward = -float('inf')

    for epoch in range(config["num_epochs"]):
        result = trainer.train()
        # Log the computed reward and portfolio value for debugging.
        computed_reward = result.get("episode_reward_mean", None)
        portfolio_value = result.get("info", {}).get("portfolio_value", None)
        print(f"Epoch: {epoch}, Reward: {computed_reward}, Portfolio Value: {portfolio_value}")

        reward_to_report = computed_reward if computed_reward is not None and computed_reward != -float('inf') else 0

        if reward_to_report > best_mean_reward:
            best_mean_reward = reward_to_report

    return session.report({"episode_reward_mean": best_mean_reward})

# ======================
# Optuna Optimization
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
        "model": {  # Model config for the FC layers
            "fcnet_hiddens": None,  # To be set below
            "fcnet_activation": trial.suggest_categorical("activation", ["relu", "tanh"])
        }
    }

    # Convert the chosen fcnet_hiddens string to a list of ints.
    fcnet_str = hyperparams["fcnet_hiddens"]
    hyperparams["model"]["fcnet_hiddens"] = [int(x) for x in fcnet_str.split(",")]
    
    # Also set recurrent model parameters (native preprocessor is used).
    hyperparams["model"]["use_lstm"] = True
    hyperparams["model"]["max_seq_len"] = ENV_BASE_CONFIG["window_size"]
    hyperparams["model"]["lstm_cell_size"] = 512

    # Train and automatically report via tune.report()
    env = gym.make('forex-v0', **ENV_BASE_CONFIG)
    trainer = train_model(hyperparams, tune_mode=True)

# ======================
# Main Execution
# ======================
def train_forex_model():
    config = {
        "logger_config": {"logdir": "logs"},
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

    # Define resources per trial.
    num_rollout_workers = 1  # Consistent with PPOConfig.
    resources_per_trial = PlacementGroupFactory(
        [{'CPU': 1.0, 'GPU': 0.0 if TUNING_SETTINGS["resource"]["num_gpus"] == 0 else 0.5}] +
        [{'CPU': 1.0, 'GPU': 0.0 if TUNING_SETTINGS["resource"]["num_gpus"] == 0 else 0.5}] * num_rollout_workers
    )

    analysis = tune.run(
        tune.with_parameters(train_model, tune_mode=False),
        config=config,
        storage_path=f"file://{os.path.abspath('logs')}",
        stop={"training_iteration": 100},
        checkpoint_at_end=False,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
        metric="episode_reward_mean",
        mode="max",
        verbose=1,
        resources_per_trial=resources_per_trial
    )

    best_checkpoint = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
        mode="max"
    )

    print(f"Best checkpoint saved at: {best_checkpoint}")

    # Save the best performing model.
    env = gym.make('forex-v0', **ENV_BASE_CONFIG)
    final_trainer = train_model(analysis.best_config, tune_mode=False)
    final_trainer.restore(best_checkpoint)
    final_trainer.save("trained_forex_model")

    print("Training completed. Best model saved.")

if __name__ == "__main__":
    # Create the environment and ensure it conforms to the Gymnasium API.
    env = gym.make('forex-v0', **ENV_BASE_CONFIG)
    check_env(env.unwrapped)

    ray.init(num_cpus=4, num_gpus=1)
    train_forex_model()
    ray.shutdown()

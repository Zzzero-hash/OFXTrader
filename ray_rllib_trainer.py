import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from forex_env import ForexEnv
import optuna

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
    "tune_epochs": 5,
    "final_epochs": 100,
    "resource": {"num_gpus": 1}
}

# ======================
# Environment Registration
# ======================
tune.register_env("forex-v0", lambda config: ForexEnv(**config))

# ======================
# Training Function
# ======================
def train_model(config, tune_mode=True):
    # Instantiate environment to retrieve RLModule spaces
    env_instance = ForexEnv(**ENV_BASE_CONFIG)
    algo_config = (
        PPOConfig()
        .api_stack(enable_env_runner_and_connector_v2=False, enable_rl_module_and_learner=False)
        .environment(env="forex-v0", env_config=ENV_BASE_CONFIG)
        .framework("torch")
        .resources(**TUNING_SETTINGS["resource"])
        .training(
            train_batch_size=config["train_batch_size"],
            lr=config["lr"],
            gamma=config["gamma"]
            # Removed model=config["model_config"]
        )
    )
    # Replace the method call with a property assignment:
    algo_config._model_config = {
        "observation_space": env_instance.observation_space,
        "action_space": env_instance.action_space,
        "inference_only": False,
        "learner_only": False,
        "model_config": config["model_config"]
    }
    
    # Updated deprecated call
    trainer = algo_config.build_algo()
    
    if tune_mode:  # Tuning mode with limited iterations
        for _ in range(config["num_epochs"]):
            result = trainer.train()
        tune.report(mean_reward=result["episode_reward_mean"])
    else:  # Full training mode
        for _ in range(config["num_epochs"]):
            trainer.train()
    
    return trainer

# ======================
# Optuna Optimization
# ======================
def objective(trial):
    hyperparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        # Changed fcnet_hiddens options to strings
        "fcnet_hiddens": trial.suggest_categorical(
            "fcnet_hiddens", ["256,256", "512,256", "512,512"]
        ),
        "train_batch_size": trial.suggest_categorical(
            "train_batch_size", [2048, 4096, 8192]
        ),
        "num_epochs": TUNING_SETTINGS["tune_epochs"],
        "model_config": {
            "fcnet_hiddens": None,  # Will be set below
            "fcnet_activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]
            )
        }
    }
    # Convert the selected string into a list of integers
    hidden_str = hyperparams.pop("fcnet_hiddens")
    hyperparams["model_config"]["fcnet_hiddens"] = [int(x) for x in hidden_str.split(",")]
    
    trainer = train_model(hyperparams, tune_mode=True)
    result = trainer.train()
    return result["episode_reward_mean"]

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Initialize Ray once
    ray.init(ignore_reinit_error=True)

    # Hyperparameter Optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TUNING_SETTINGS["n_trials"])

    # Final Training with Best Params
    best_config = {
        "train_batch_size": study.best_params["train_batch_size"],
        "lr": study.best_params["lr"],
        "gamma": study.best_params["gamma"],
        "num_epochs": TUNING_SETTINGS["final_epochs"],
        "model_config": {
            # Convert best_params string choice to a list of ints
            "fcnet_hiddens": [int(x) for x in study.best_params["fcnet_hiddens"].split(",")],
            "fcnet_activation": study.best_params["activation"]
        }
    }

    final_trainer = train_model(best_config)
    final_trainer.save("trained_forex_model")
    ray.shutdown()
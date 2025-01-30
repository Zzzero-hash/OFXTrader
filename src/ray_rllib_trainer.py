import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.forex_env import ForexEnv

def env_creator(env_config):
    return ForexEnv(
        instrument=env_config["instrument"],
        start_date=env_config["start_date"],
        end_date=env_config["end_date"],
        granularity=env_config["granularity"],
        window_size=env_config["window_size"],
        initial_balance=env_config.get("initial_balance", 1000),
        leverage=env_config.get("leverage", 50)
    )

register_env("forex-env", env_creator)

def train_model(config):
    ray.init(ignore_reinit_error=True)
    algo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="forex-env", env_config=config["env_config"])
        .framework("torch")
        .resources(num_gpus=0)
        .training(
            train_batch_size=config.get("train_batch_size", 4000),
            lr=config.get("lr", 1e-4),
            gamma=config.get("gamma", 0.99),
            model=config.get("model_config", {"fcnet_hiddens": [256, 256],
                                              "fcnet_activation": "relu"})
        )
    )
    trainer = algo_config.build()
    for _ in range(config.get("num_epochs", 100)):
        result = trainer.train()
        # Print all keys in result to identify available metrics
        print("Result dictionary keys:", list(result.keys()))
        
        if 'episode_reward' in result:  # Check for relevant metric key(s)
            episode_rewards = result['episode_reward']
            avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
            print(f"Average Episode Reward: {avg_episode_reward}")
        else:
            print("No episode reward metrics available yet.")
        
        tune.report(
            episode_reward_mean=avg_episode_reward if 'episode_reward' in result else None,
            episodes_total=result.get('episodes_total', 0),
            info=result
        )
    ray.shutdown()
    return trainer

if __name__ == "__main__":
    config = {
        "env_config": {
            "instrument": "EUR_USD",
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "granularity": "M1",
            "window_size": 24,
            "initial_balance": 10000,
            "leverage": 50,
            "count": 5000
        },
        "num_gpus": 1,
        "train_batch_size": 4000,
        "lr": 3e-4,
        "gamma": 0.95,
        "num_epochs": 100,
        "use_lstm": True,
        "model_config": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "tanh"
        }
    }
    trainer = train_model(config)
    trainer.save("trained_forex_model")
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import yaml
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as sb3_logger_configure
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

VALID_MODEL_NAMES = ["ppo", "sac", "td3"]


def read_yaml_file(file_path: str) -> dict:
    """Reads yaml file from file path and returns dict."""
    with open(file_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def get_model_config(library: str, model: str) -> dict:
    config_file = f"./config/{str.lower(library)}.yml"
    return read_yaml_file(config_file)[model]


@runtime_checkable
class Trainer(Protocol):
    """
    Protocol to train a library-independent RL algorithm on a gym environment.
    """

    model: str
    env_name: str
    n_envs: int
    total_timesteps: int
    library: str
    log: bool

    def _init_agent(self) -> None:
        """Initialize the agent."""

    def train(self, total_timesteps) -> None:
        """Train agent on environment for total_timesteps episdodes."""


@dataclass
class SB3Trainer:
    model: str = "ppo"
    env_name: str = "CartPole-v1"
    n_envs: int = 1
    total_timesteps: int = 10_000
    library: str = "SB3"
    log: bool = False

    def __post_init__(self) -> None:
        if self.model not in VALID_MODEL_NAMES:
            raise ValueError(f"model must be one of: {VALID_MODEL_NAMES}")

        if self.n_envs <= 0:
            raise ValueError("n_envs must be greater than zero.")

        self.config = self._get_config()
        self._init_agent()

    def _get_config(self) -> dict:
        config = get_model_config(self.library, self.model)
        config.update({"env": self._get_env()})
        return config

    def _get_env(self):
        return make_vec_env(self.env_name, self.n_envs, vec_env_cls=SubprocVecEnv)

    def _get_model_class(self):
        match self.model:
            case "ppo":
                from stable_baselines3 import PPO as ModelClass
            case "sac":
                from stable_baselines3 import SAC as ModelClass
            case "td3":
                from stable_baselines3 import TD3 as ModelClass
        return ModelClass

    def _init_agent(self) -> None:
        ModelClass = self._get_model_class()
        self.agent = ModelClass(**self.config)

        if self.log:
            self._setup_logger()

    def _setup_logger(self) -> None:
        log_path = f"./logs/{str.lower(self.library)}"
        logger = sb3_logger_configure(log_path, ["stdout", "tensorboard"])
        self.agent.set_logger(logger)

    def train(self) -> None:
        _ = self.agent.learn(self.total_timesteps)


@dataclass
class RLlibTrainer:
    model: str = "ppo"
    env_name: str = "CartPole-v1"
    n_envs: int = 1
    total_timesteps: int = 10_000
    library: str = "RLlib"
    log: bool = False

    def __post_init__(self) -> None:
        if self.model not in VALID_MODEL_NAMES:
            raise ValueError(f"model must be one of: {VALID_MODEL_NAMES}")

        if (self.n_envs <= 0) or (self.n_envs > os.cpu_count()):
            raise ValueError("n_envs must be between 1 and number of CPU cores.")

        self.config = self._get_config()
        self._init_agent()

    def _get_config(self):
        config = get_model_config(self.library, self.model)

        # update for vectorized envs, i.e. if n_envs > 1
        config.update({"num_workers": self.n_envs})
        config.update({"train_batch_size": config["train_batch_size"] * self.n_envs})

        # setup tesnorboard logging
        if self.log:
            config["logger_config"] = {
                "type": "ray.tune.logger.TBXLogger",
                "logdir": f"./logs/{str.lower(self.library)}",
            }

        return config

    def _get_model_class(self):
        match self.model:
            case "ppo":
                from ray.rllib.algorithms.ppo import PPO as ModelClass
            case "sac":
                from ray.rllib.algorithms.sac import SAC as ModelClass
            case "td3":
                from ray.rllib.algorithms.td3 import TD3 as ModelClass
        return ModelClass

    def _init_agent(self) -> None:
        ModelClass = self._get_model_class()
        self.agent = ModelClass(config=self.config, env=self.env_name)

    def train(self) -> None:
        timesteps = 0
        while timesteps < self.total_timesteps:
            result = self.agent.train()
            timesteps = result["timesteps_total"]


if __name__ == "__main__":  # pragma: no cover
    print("Starting training")

    rllib_trainer = RLlibTrainer()
    rllib_trainer.train()

    sb3_trainer = SB3Trainer()
    sb3_trainer.train()

    print("Finished training both libraries")

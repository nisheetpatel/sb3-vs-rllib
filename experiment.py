import itertools
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from trainer import RLlibTrainer, SB3Trainer, Trainer


@dataclass
class ExperimentConfig:
    models: list[str] = field(default_factory=lambda: ["ppo", "sac", "td3"])
    envs: list[str] = field(
        default_factory=lambda: [
            "CartPole-v1",
            "LunarLander-v2",
            "Pendulum-v1",
            "LunarLanderContinuous-v2",
            "BipedalWalker-v3",
        ]
    )
    timesteps: list[int] = field(default_factory=lambda: [1_000, 10_000, 100_000])
    n_envs: int = 1
    libraries: list[str] = field(default_factory=lambda: ["sb3", "rllib"])

    def __post_init__(self) -> None:
        self.model_env_combos = [
            c
            for c in itertools.product(self.models, self.envs)
            if c in self.supported_model_env_combinations()
        ]

        for lib_name in self.libraries:
            assert lib_name in ["sb3", "rllib"], "Invalid library name specified."

        lib_to_trainer = {"sb3": SB3Trainer, "rllib": RLlibTrainer}
        self.trainer_classes = [lib_to_trainer[lib] for lib in self.libraries]

    @staticmethod
    def supported_model_env_combinations() -> list[tuple]:
        return [
            ("ppo", "CartPole-v1"),
            ("ppo", "LunarLander-v2"),
            ("ppo", "BipedalWalker-v3"),
            ("sac", "Pendulum-v1"),
            ("sac", "LunarLanderContinuous-v2"),
            ("sac", "BipedalWalker-v3"),
            ("td3", "Pendulum-v1"),
            ("td3", "LunarLanderContinuous-v2"),
            ("td3", "BipedalWalker-v3"),
        ]


@dataclass
class DataHandler:
    df: pd.DataFrame = pd.DataFrame(
        columns=[
            "Library",
            "Model",
            "Environment",
            "n_envs",
            "Timesteps",
            "Training time",
        ]
    )
    save_path: str = "./logs/"

    def add_datapoint(self, trainer: Trainer, training_time: float) -> None:
        row = [
            trainer.library,
            trainer.model,
            trainer.env_name,
            trainer.n_envs,
            trainer.total_timesteps,
            training_time,
        ]
        self.df.loc[-1] = row
        self.df.index += 1
        self.df.sort_index(inplace=True)

    def save_data(self, save_name="training_time") -> None:
        file = Path(self.save_path + save_name + ".csv")
        if file.is_file():
            df = pd.read_csv(file)
            self.df = pd.concat([df, self.df], ignore_index=True)
        self.df.to_csv(file, index=False)

    def plot(self, save_name="training_time") -> None:
        sns.set()
        # df_plot = self.df.loc[self.df["Timesteps"] == total_timesteps]
        g = sns.catplot(
            data=self.df,
            kind="bar",
            x="Environment",
            y="Training time",
            hue="Library",
            col="Timesteps",
            row="Model",
            palette=sns.color_palette(n_colors=2),
            sharex=False,
            sharey=False,
        )
        g.set_axis_labels("", "Training time (seconds)")
        plt.savefig(self.save_path + save_name + ".svg")
        plt.savefig(self.save_path + save_name + ".png")
        plt.close()


@dataclass
class Experiment:
    config: ExperimentConfig = ExperimentConfig()
    data_handler: DataHandler = DataHandler()

    def _time(self, trainer: Trainer) -> float:
        start = time()
        _ = trainer.train()
        return time() - start

    def _run_single(self, trainer_class: Trainer, model, env, n_envs, timesteps):
        trainer = trainer_class(model, env, n_envs, timesteps)
        time_taken = self._time(trainer)
        self.data_handler.add_datapoint(trainer, time_taken)

    def run(self) -> None:
        n_envs = self.config.n_envs
        for timesteps in self.config.timesteps:
            for model, env in self.config.model_env_combos:
                for trainer_class in self.config.trainer_classes:
                    self._run_single(trainer_class, model, env, n_envs, timesteps)

    def save_data_and_figures(self, save_name="training_time") -> None:
        self.data_handler.save_data(save_name)
        self.data_handler.plot(save_name)


if __name__ == "__main__":  # pragma: no cover
    config = ExperimentConfig(
        models=["ppo", "sac"],
        envs=["CartPole-v1", "BipedalWalker-v3"],
        timesteps=[2_000, 4_000],
    )

    experiment = Experiment(config=config)
    experiment.run()
    experiment.save_data_and_figures()

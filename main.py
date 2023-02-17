from experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        # using defaults to train all valid models and envs
        # models=["ppo", "sac"],    # set specific ones like this
        # envs=["BipedalWalker-v3"],
        timesteps=[16_000, 64_000, 128_000],
        n_envs=12,  # set n_envs = 1 for non-vectorized
        libraries=["sb3", "rllib"],
    )

    experiment = Experiment(config=config)
    experiment.run()
    experiment.save_data_and_figures(save_name="training_time_vec")


if __name__ == "__main__":
    main()

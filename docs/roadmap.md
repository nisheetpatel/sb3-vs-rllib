## Roadmap

### Setup the repository and environment

- [x] Install all packages
- [x] Setup gitignore, linter specifications, vscode settings

### Develop algorithm testing pipeline

- [x] Run single algorithm (PPO) from both libraries on single task (CartPole)
  - [x] SB3
  - [x] RLlib
- [x] Setup training, testing, and logging
  - [x] SB3
  - [x] RLlib

### Run pipeline for multiple algorithms & environments

#### Non-vectorized environments

##### Benchmarking time taken for training

_Took x seconds for training y episodes_

- [x] RL algorithms: SAC, PPO, TD3
  - [x] Ensure that hyperparameters are fixed for v0
    - [x] Manually set the hyperparameters to be the same for both libraries. This will ensure a fair comparison of training time.
- [x] Environments:
  - [x] PPO: CartPole-v1, LunarLander-v2, BipedalWalker-v3
  - [x] SAC, TD3: Pendulum-v1, LunarLanderContinuous-v2, BipedalWalker-v3
  - [x] Non-vectorized environments

Evalutation and logging:

- [x] Dataframes (Library, Model, Environment, Timesteps, Training time)
- [x] Plots (Time taken for training vs environment-model-library combination)

##### Benchmarking speed of convergence

_Took x episodes to reach max reward_

- [ ] Get best hypperparmeters for each environment from the respective libraries

We decided that this is unnecessary for two reasons:

1. Given the correct hyperparameters, as long as they are the same across both libraries, the models should converge in roughly the same number of timesteps. So doing this would not give us any information about which library is better to use in terms of training time.
2. This might be better done with hyperparameter tuning, which is a project in and of itself. There might be a master's student joining who would do this.

Evaluation and logging:

- [x] Tensorboard

#### Vectorized environments

- [x] Now supported for both libraries with everything as in Non-vectorized environments.

### Tests

- [x] Tests written in `tests/` directory
- [x] Coverage reports available in `docs/test_coverage_reports/`
  - Currently covering pretty much everyhing other than the code in `if __name__ == '__main__':`

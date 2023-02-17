# Benchmarking RL libraries

In this mini-project, I compare and benchmark the performance of some RL algorithms from two popular libraries, [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3/) & [RLlib](https://github.com/ray-project/ray/tree/master/rllib). View the full roadmap [here](docs/roadmap.md). If you would like to view my notes on the experience of setting up these libraries, see [this document](docs/notes_sb3_vs_rllib.md).

## Installation

Note: _I use [mamba](https://github.com/mamba-org/vscode-micromamba) as my reference manager of choice. It is way faster than conda and basically a drop-in replacement (replace `conda` with `mamba`)._

Installation is fairly simple:

```sh
# clone this repository and change directory the repository directory
git clone https://github.com/nisheetpatel/sb3-vs-rllib.git
cd sb3-vs-rllib/

# create and activate environment (use conda instead of mamba if you like)
mamba create --name sb3-vs-rllib
mamba activate sb3-vs-rllib

# install all required packages
mamba install pip
pip install stable-baselines3[extra] ray[rllib] gym free-mujoco-py pyglet yaml swig box2d box2d-kengz seaborn
```

Note: If `free-mujoco-py` gives you trouble, look at [the official troubleshooting steps for Ubuntu](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting).

## Usage

### Benchmarking experiments

`python main.py` runs an example experiment with parameters specified as the ExperimentConfig arguments. For training with non-vectorized environments, specify `n_envs = 1`, otherwise you can specify `n_envs` up to one less than the number of logical cores available on your machine.

#### Logs and results

- Figures, dataframes, and tensorboard logs will be saved in the `logs` folder.
- For tensorboard tracking, run `python -m tensorboard.main --logdir ./logs`.

#### Model configuration

Currently, the configuration files are present in the _config_ directory where the algorithms from both libraries are initialized with the same set of parameters (as far as possible) to ensure a relatively fair comparison.

## Tests

Install required packages and generate coverage reports (currently covering 99%).

### Packages required

```sh
pip install pytest pytest-cov hypothesis
```

### Running the tests

```sh
pytest --cov --cov-report=html
```

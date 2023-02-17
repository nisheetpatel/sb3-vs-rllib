import os
import time
from dataclasses import dataclass

import pytest
from experiment import DataHandler, Experiment, ExperimentConfig


@dataclass
class DummyTrainer:
    library: str = "foo"
    model: str = "bar"
    env_name: str = "baz"
    n_envs: int = 1
    total_timesteps: int = 0

    @staticmethod
    def train() -> None:
        time.sleep(0.01)


@pytest.fixture(name="expt")
def experiment_fixture() -> Experiment:
    config = ExperimentConfig(timesteps=[0])
    data_handler = DataHandler(save_path="./")
    return Experiment(config, data_handler)


def test_data_handler_adds_dummy_datapoint() -> None:
    data_handler = DataHandler()
    # len_df = len(data_handler.df)
    data_handler.add_datapoint(DummyTrainer(), training_time=4.0)
    assert len(data_handler.df) == 1


def test_data_and_figure_saving(expt: Experiment) -> None:
    expt.data_handler.add_datapoint(DummyTrainer(), 4)
    expt.save_data_and_figures("dummy_c8s03jvg01")

    # test data appending and saving works
    expt.data_handler.add_datapoint(DummyTrainer(), 2)
    expt.save_data_and_figures("dummy_c8s03jvg01")

    for file_suffix in [".csv", ".png", ".svg"]:
        assert os.path.exists("./dummy_c8s03jvg01" + file_suffix)
        os.remove("./dummy_c8s03jvg01" + file_suffix)


def test_experiment_timer_function(expt: Experiment):
    time_taken = expt._time(DummyTrainer())  # pylint: disable=protected-access
    assert 0 < time_taken < 0.02


def test_experiment_runs(expt: Experiment) -> None:
    expt.run()

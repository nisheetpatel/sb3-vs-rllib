import os

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, text
from trainer import RLlibTrainer, SB3Trainer, Trainer

# initializing valid trainers, model, and envs
TRAINER_CLASSES = [SB3Trainer, RLlibTrainer]
VALID_MODEL_NAMES = ["ppo", "sac", "td3"]
COMMON_ENV_NAME = "BipedalWalker-v3"


@pytest.mark.parametrize("lib_trainer", TRAINER_CLASSES)
def test_trainers_follow_protocol(lib_trainer) -> None:
    assert isinstance(lib_trainer(), Trainer)


@pytest.mark.parametrize("lib_trainer", TRAINER_CLASSES)
@given(integers(max_value=0))
@settings(max_examples=1)
def test_trainers_min_n_envs(lib_trainer, n_envs: int):
    with pytest.raises(ValueError):
        lib_trainer(n_envs=n_envs)


@given(integers(min_value=os.cpu_count() + 1))
@settings(max_examples=1)
def test_trainers_max_n_envs(n_envs: int):
    with pytest.raises(ValueError):
        RLlibTrainer(n_envs=n_envs)


@pytest.mark.parametrize("lib_trainer", TRAINER_CLASSES)
@given(text())
@settings(max_examples=5)
def test_trainers_with_invalid_models(lib_trainer, model: str):
    if model not in VALID_MODEL_NAMES:
        with pytest.raises(ValueError):
            lib_trainer(model=model)


@pytest.mark.parametrize("lib_trainer", TRAINER_CLASSES)
@pytest.mark.parametrize("model", VALID_MODEL_NAMES)
def test_trainers_with_valid_models(lib_trainer, model):
    lib_trainer(model=model, env_name=COMMON_ENV_NAME)


def test_sb3_tensorboard_logging() -> None:
    trainer = SB3Trainer(log=True)
    assert trainer.agent.logger is not None


def test_rllib_logging() -> None:
    trainer = RLlibTrainer(log=True)
    assert isinstance(trainer.config["logger_config"], dict)

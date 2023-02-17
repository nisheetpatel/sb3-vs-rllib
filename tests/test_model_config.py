import os.path

import pytest
from trainer import get_model_config, read_yaml_file


@pytest.mark.parametrize("lib", ["sb3", "rllib"])
def test_config_exists(lib: str) -> None:
    assert os.path.exists(f"config/{lib}.yml")


@pytest.mark.parametrize("lib", ["sb3", "rllib"])
def test_config_is_dict(lib: str) -> None:
    config = read_yaml_file(f"config/{lib}.yml")
    assert isinstance(config, dict)


@pytest.mark.parametrize("lib", ["sb3", "rllib"])
@pytest.mark.parametrize("model", ["ppo", "sac", "td3"])
def test_model_config_is_dict(lib: str, model: str) -> None:
    config = get_model_config(lib, model)
    assert isinstance(config, dict)

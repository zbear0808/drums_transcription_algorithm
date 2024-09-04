"""Setting class."""
from pathlib import Path
import dataclasses
from pydantic import BaseSettings
from typing import Dict, Any
import json

file_name = Path(__file__)


def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    Loads the settings from a JSON file on the repo root

    :return: Dictionary of settings
    """
    encoding = settings.__config__.env_file_encoding
    path = file_name.parent / "config.json"

    return json.loads(path.read_text(encoding))


class Settings(BaseSettings):
    """The base and serialized application settings"""

    class Config:
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )

    sampling_frequency: float
    mffc_our_implementation: bool
    hop_size_onset: int
    analysis_window_s: float
    n_mfcc_features: int
    FFT_hop_size: int
    threshold_prediction: float
    batch_size: int
    n_epoch: int
    dropout: float
    # TODO: hyperparameters also should come here

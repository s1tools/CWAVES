"""Configuration loading for the cwaves library.

The default configuration (``cwaves/default_config.yml``) holds the ``cwave_*``
parameters of :func:`cwaves.compute_cwave_parameters`. The ``cwave`` block of the
file can be passed directly as keyword arguments :

    from cwaves import compute_cwave_parameters, get_default_config
    cwave = get_default_config()["cwave"]
    cwave_params = compute_cwave_parameters(xs, **cwave)
"""

from pathlib import Path
from typing import Any

import yaml

#: path of the packaged default configuration file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a configuration file and return its content as a dictionary.

    Args:
        config_path: path to a YAML configuration file. If ``None``, the packaged
            ``default_config.yml`` is used.

    Returns:
        A dictionary with the configuration content (top-level keys).
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration file {path} must contain a mapping")
    return loaded


def get_default_config() -> dict[str, Any]:
    """Return the packaged default configuration as a dictionary.

    Returns:
        A dictionary with the default configuration content.
    """
    return load_config(DEFAULT_CONFIG_PATH)

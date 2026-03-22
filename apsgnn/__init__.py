"""APSGNN experiment package.

Keep config helpers importable without pulling in the full torch stack.
"""

from apsgnn.config import ExperimentConfig, load_config

__all__ = ["APSGNNModel", "ExperimentConfig", "load_config"]


def __getattr__(name: str):
    if name == "APSGNNModel":
        from apsgnn.model import APSGNNModel

        return APSGNNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

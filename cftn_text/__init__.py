"""CFTN-Text: persistent generalist and specialist towers with gated messages."""

from .config import load_config
from .model import CFTNTextModel

__all__ = ["CFTNTextModel", "load_config"]

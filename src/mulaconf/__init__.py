"""
MuLaConf
========

A Scikit-Learn-compatible Python package for Inductive Conformal Prediction (ICP)
in multi-label classification tasks using structural penalties.
"""

from .icp_wrapper import ICPWrapper
from .icp_predictor import InductiveConformalPredictor
from .prediction_regions import PredictionRegions
from . import constants

__version__ = "0.3.0"

__all__ = [
    "ICPWrapper",
    "InductiveConformalPredictor",
    "PredictionRegions",
    "constants",
    "__version__"
]

"""
CyTOF Helper Package

This package provides utilities for CyTOF data analysis, including:
- Normalization (cytof_helper.normalization)
- Statistical analysis (cytof_helper.stats)
- Visualization (cytof_helper.plotting)
- Interactive labelling (cytof_helper.interactive)
- Data smoothing (cytof_helper.smoothing)
- General utilities (cytof_helper.utils)
"""

from . import stats
from . import plotting
from . import normalization
from . import smoothing
from . import interactive
from . import utils

__version__ = "0.1.0"

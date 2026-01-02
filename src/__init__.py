"""
Source code package for the Peru Production Analysis project.
Contains all the Python code for data processing and analysis.
"""

import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import commonly used functions
from .functions.plot_production_groups import plot_production_by_group 
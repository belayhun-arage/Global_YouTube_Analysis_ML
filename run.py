# run.py

import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Now import your DataProcessor class and other components from your package
from src.data_processing import DataProcessor

# Instantiate the project
from config import data_path
processor = DataProcessor(data_path)

# Perform any further operations here

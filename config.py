# config.py
import os

absolute_path = os.path.abspath(os.path.dirname(__file__))
relative_path = "data/Global_YouTube_Statistics_2023_pruned.csv"
data_path = os.path.join(absolute_path, relative_path)
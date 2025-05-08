import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import json
import os

def create_directories():
    """Create necessary directories for the project"""
    dirs = ['models', 'logs', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def plot_training_progress(log_file: str):
    """Plot training metrics from tensorboard log file"""
    # This is a placeholder - you'll need tensorboard integration
    pass

def save_metrics(metrics: Dict, filename: str):
    """Save evaluation metrics to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filename: str) -> Dict:
    """Load evaluation metrics from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_uncertainty_threshold(uncertainties: np.ndarray, percentile: float = 95) -> float:
    """Calculate uncertainty threshold based on collected data"""
    return np.percentile(uncertainties, percentile)

def normalize_observation(obs: np.ndarray) -> np.ndarray:
    """Normalize observation values to [-1, 1] range"""
    return np.clip(obs / 100.0, -1, 1)  # Assuming max values around ±100
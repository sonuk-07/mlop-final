# pipelines/hyperparameter.py
import random
import logging

log = logging.getLogger(__name__)

def prepare_hyperparameters() -> dict:
    """
    Randomly prepare hyperparameters (for demonstration or optuna later).
    Returns Redis-friendly dict.
    """
    hyperparams = {
        "iterations": random.choice([100, 200, 300]),
        "depth": random.choice([4, 6, 8]),
        "learning_rate": random.choice([0.01, 0.05, 0.1]),
        "l2_leaf_reg": random.choice([3, 5, 7])
    }
    log.info(f"Hyperparameters prepared: {hyperparams}")
    return hyperparams

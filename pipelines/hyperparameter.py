import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import logging
import pandas as pd

log = logging.getLogger(__name__)

# Assuming X_train and y_train are already prepared elsewhere
# You can pass them to this function or load from Redis if needed

def tune_catboost(X_train_res, y_train_res, n_trials=30):
    """
    Optimize CatBoost hyperparameters using Optuna.
    Returns best hyperparameters as a dict.
    """
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5),
        }
        model = CatBoostClassifier(**params, random_state=42, verbose=0)
        score = cross_val_score(model, X_train_res, y_train_res, cv=3, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    log.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params

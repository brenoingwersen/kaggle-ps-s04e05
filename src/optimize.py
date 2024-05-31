import importlib
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import optuna
from optuna import trial
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from logging_config import logging
from config import (TRAIN_FILE, TARGET_COL, ID_COL,
                    MODEL_TYPE, FEATURE_ENGINEER,
                    PREPROCESS, FEATURE_SELECTION,
                    METRIC_NAME, N_FOLDS, SEED,
                    OPTUNA_STORAGE, OPTUNA_STUDY_NAME,
                    OPTUNA_DIRECTION, OPTUNA_N_TRIALS,
                    SAMPLER_TYPE, SAMPLER_PARAMS,
                    PRUNER_TYPE, PRUNER_PARAMS)


# Configs logger and warnings
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Import training data
train_data = pd.read_csv(TRAIN_FILE)

# Build the pipeline
feature_engineer = importlib.import_module(
    FEATURE_ENGINEER.replace("/", ".").removesuffix(".py")).feature_engineer
preprocessor = importlib.import_module(
    PREPROCESS.replace("/", ".").removesuffix(".py")).preprocessor
feature_selector = importlib.import_module(
    FEATURE_SELECTION.replace("/", ".").removesuffix(".py")).feature_selector


def objective_function(trial):
    if MODEL_TYPE == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "n_jobs": 20,
            "verbose": -1
        }
        model = XGBRegressor(**params)
    elif MODEL_TYPE == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "verbose": -1
        }
        model = LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    pipeline = Pipeline(steps=[
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("model", model)
    ])

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(estimator=pipeline,
                             X=train_data.drop(columns=[ID_COL, TARGET_COL]),
                             y=train_data[TARGET_COL],
                             cv=cv,
                             scoring=METRIC_NAME)
    mean_score = np.mean(scores)
    return mean_score


########################################
# TQDM + OPTUNA
########################################
class ProgressBar(tqdm):
    def update_to(self, study, trial):
        best_trial_score = 0.
        best_trial_number = 0
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value > best_trial_score:
                best_trial_score = trial.value
                best_trial_number = trial.number

        self.set_description(
            f"Best trial: {best_trial_number} - Best score: {best_trial_score:.5f}")
        self.update(1)


########################################
# OPTUNA
########################################
# Sampler
if SAMPLER_TYPE is not None:
    Sampler = getattr(optuna.samplers, SAMPLER_TYPE)
    sampler = Sampler(**(SAMPLER_PARAMS or {}))
else:
    sampler = None

# Pruner
if PRUNER_TYPE is not None:
    Pruner = getattr(optuna.pruners, PRUNER_TYPE)
    pruner = Pruner(**(PRUNER_PARAMS or {}))
else:
    pruner = None

study = optuna.create_study(
    storage=OPTUNA_STORAGE,
    study_name=OPTUNA_STUDY_NAME,
    direction=OPTUNA_DIRECTION,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner
)

logger.info("Start optimizing...")
with ProgressBar(total=OPTUNA_N_TRIALS) as prog_bar:
    study.optimize(objective_function,
                   n_trials=OPTUNA_N_TRIALS,
                   show_progress_bar=False,
                   callbacks=[prog_bar.update_to])
logger.info("Optimization complete!")
logger.info(
    f"Best trial: {study.best_trial.number} - Score: {study.best_trial.value:.5f}")

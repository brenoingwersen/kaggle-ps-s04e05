import os
import yaml
import re
from uuid import uuid4


with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

    BUILD_INFO = CONFIG.get("build")
    BUILD_DIR = BUILD_INFO.get("dir")

    # Create the build directory
    if not os.path.exists(BUILD_DIR):
        os.mkdir(BUILD_DIR)

    # Data
    DATA_CONFIG = CONFIG.get("data")
    DATA_PATH = DATA_CONFIG.get("path")
    TRAIN = DATA_CONFIG.get("train")
    TEST = DATA_CONFIG.get("test")
    TRAIN_FILE = os.path.join(DATA_PATH, TRAIN)
    TEST_FILE = os.path.join(DATA_PATH, TEST)
    ID_COL = DATA_CONFIG.get("id_col")
    TARGET_COL = DATA_CONFIG.get("target_col")

    # Pipeline
    FEATURE_ENGINEER = CONFIG.get("feature_engineering").get("path")
    PREPROCESS = CONFIG.get("preprocess").get("path")
    FEATURE_SELECTION = CONFIG.get("feature_selection").get("path")

    # Model
    MODEL = CONFIG.get("model")
    MODEL_TYPE = MODEL.get("type")
    MODEL_PARAMS = MODEL.get("parameters")

    # Metrics
    METRIC_NAME = CONFIG.get("metrics").get("name")

    # Cross-validation
    N_FOLDS = CONFIG.get("cross_validation").get("n_folds")
    SEED = CONFIG.get("cross_validation").get("random_state")

    # Optuna
    OPTUNA_CONFIG = CONFIG.get("optuna")
    OPTUNA_STUDY_NAME = OPTUNA_CONFIG.get("study_name")
    OPTUNA_DIRECTION = OPTUNA_CONFIG.get("direction")
    OPTUNA_N_TRIALS = OPTUNA_CONFIG.get("n_trials")
    # Sampler
    SAMPLER_TYPE = OPTUNA_CONFIG.get("sampler").get("type")
    SAMPLER_PARAMS = OPTUNA_CONFIG.get("sampler").get("parameters")
    # Pruner
    PRUNER_TYPE = OPTUNA_CONFIG.get("pruner").get("type")
    PRUNER_PARAMS = OPTUNA_CONFIG.get("pruner").get("parameters")

    if OPTUNA_CONFIG.get("storage") == "sqlite":
        study_name_fmt = re.sub(pattern=r"\s+",
                                repl="_",
                                string=OPTUNA_STUDY_NAME.lower()).strip()
        OPTUNA_STORAGE = f"sqlite:///{BUILD_DIR}/{study_name_fmt}.db"

import os
import yaml
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

    # Model
    MODEL = CONFIG.get("model")
    MODEL_TYPE = MODEL.get("type")
    MODEL_PARAMS = MODEL.get("parameters")

    # Metrics
    METRIC_NAME = CONFIG.get("metrics").get("name")

    # Cross-validation
    N_FOLDS = CONFIG.get("cross_validation").get("n_folds")
    SEED = CONFIG.get("cross_validation").get("random_state")

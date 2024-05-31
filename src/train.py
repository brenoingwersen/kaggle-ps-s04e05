import dill
import yaml
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from config import BUILD_DIR, TRAIN_FILE, TEST_FILE, ID_COL, TRAIN_PARAMS, TARGET_COL, METRIC_NAME, N_FOLDS, SEED
from logging_config import logging


logger = logging.getLogger(__name__)


def build_pipeline(BUILD_DIR):
    # Feature engineer
    with open(f"{BUILD_DIR}/feature_engineer.joblib", "rb") as io:
        feature_engineer = dill.load(io)

    # Preprocessor
    with open(f"{BUILD_DIR}/preprocessor.joblib", "rb") as io:
        preprocessor = dill.load(io)

    # Feature selection
    with open(f"{BUILD_DIR}/feature_selector.joblib", "rb") as io:
        feature_selector = dill.load(io)

    # Model
    with open(f"{BUILD_DIR}/model.joblib", "rb") as io:
        model = dill.load(io)

    pipe = Pipeline(steps=[
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("model", model)
    ])

    return pipe


def train(pipeline, train_data, test_data):
    """
    """
    X = train_data.drop(columns=[ID_COL, TARGET_COL])
    y = train_data[TARGET_COL]

    X_test = test_data.drop(columns=[ID_COL])

    # Cross validator
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Metric
    scoring_func = get_scorer(METRIC_NAME)

    scores = []
    test_preds = np.zeros(len(X_test))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Train/validation split
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        # Train the model
        start_time = perf_counter()
        pipeline.fit(X_train, y_train, **(TRAIN_PARAMS or {}))
        end_time = perf_counter()
        train_time = end_time - start_time
        # Calculate fold score
        fold_score = scoring_func(pipeline, X_val, y_val)
        logger.info(
            f"Fold {fold_idx + 1}/{N_FOLDS} - {METRIC_NAME} score: {fold_score:.5f} - train time: {train_time:,.0f} s")
        scores.append(fold_score)
        # Run test predictions
        y_test_pred = pipeline.predict(X_test)
        test_preds += y_test_pred/N_FOLDS

    return np.mean(scores), test_preds


if __name__ == "__main__":
    # Instantiate the pipeline
    pipeline = build_pipeline(BUILD_DIR)
    logger.info("Pipeline created.")

    # Load the data
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    logger.info("Train/Test data loaded.")

    # Train the model
    start_time = perf_counter()
    cv_score, test_preds = train(pipeline, train_data, test_data)
    end_time = perf_counter()
    train_time = end_time - start_time

    # Store build results
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)

    CONFIG.update(
        {"results": {"cv_score": float(cv_score),
                     "train_time": float(train_time)}})
    with open(f"{BUILD_DIR}/config.yaml", "w") as f:
        yaml.safe_dump(CONFIG, f)
    logger.info("Training results stored.")

    # Store predictions for submission
    test_data[TARGET_COL] = test_preds
    submit_df = test_data[[ID_COL, TARGET_COL]].copy(deep=True)
    submit_df.to_csv(f"{BUILD_DIR}/submission.csv", index=False)
    logger.info("Submission results stored.")

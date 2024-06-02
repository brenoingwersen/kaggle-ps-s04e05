import importlib
import dill
# Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# Configs
from config import BUILD_DIR, MODEL_TYPE, MODEL_PARAMS


# Imports the specified model
models = {
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
    "catboost": CatBoostRegressor
}

if models.get(MODEL_TYPE) is None:
    raise ValueError(
        f"Model type {MODEL_TYPE} not supported. Available types: {', '.join(models.keys())}")
else:
    model = models.get(MODEL_TYPE)(**(MODEL_PARAMS or {}))

if __name__ == "__main__":
    # Store the model
    with open(f"{BUILD_DIR}/model.joblib", "wb") as io:
        dill.dump(model, io)

from catboost import CatBoostRegressor
from src.config import MODEL_PARAMS


model = CatBoostRegressor(**MODEL_PARAMS)

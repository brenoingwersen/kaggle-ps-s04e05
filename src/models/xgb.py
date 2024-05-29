from xgboost import XGBRegressor
from src.config import MODEL_PARAMS


model = XGBRegressor(**MODEL_PARAMS)

from lightgbm import LGBMRegressor
from src.config import MODEL_PARAMS


model = LGBMRegressor(**MODEL_PARAMS)

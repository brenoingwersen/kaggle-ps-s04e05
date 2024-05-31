import dill
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from src.config import BUILD_DIR


def create_features(s):
    """
    """
    s_copy = s.copy(deep=True)
    s_copy["fsum"] = s.sum(axis=1)
    s_copy["special1"] = s_copy["fsum"].isin(np.arange(72, 76))
    return s_copy


feature_engineer = FunctionTransformer(create_features)

if __name__ == "__main__":
    with open(f"{BUILD_DIR}/feature_engineer.joblib", "wb") as io:
        dill.dump(feature_engineer, io)

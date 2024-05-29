import dill
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from config import BUILD_DIR


def create_features(s: pd.Series):
    """
    Return new features
    """
    s_copy = s.copy(deep=True)
    s_copy["ClimateAnthropogenicInteraction"] = (s["MonsoonIntensity"] + s["ClimateChange"]) * (
        s["Deforestation"] + s["Urbanization"] + s["AgriculturalPractices"] + s["Encroachments"])
    s_copy["InfrastructurePreventionInteraction"] = (s["DamsQuality"] + s["DrainageSystems"] + s["DeterioratingInfrastructure"]) * (
        s["RiverManagement"] + s["IneffectiveDisasterPreparedness"] + s["InadequatePlanning"])
    s_copy["sum"] = s.sum(axis=1)
    s_copy["std"] = s.std(axis=1)
    s_copy["mean"] = s.mean(axis=1)
    s_copy["max"] = s.max(axis=1)
    s_copy["min"] = s.min(axis=1)
    s_copy["mode"] = s.mode(axis=1)[0]
    s_copy["median"] = s.median(axis=1)
    s_copy["q_25th"] = s.quantile(0.25, axis=1)
    s_copy["q_75th"] = s.quantile(0.75, axis=1)
    s_copy["skew"] = s.skew(axis=1)
    s_copy["kurt"] = s.kurt(axis=1)
    s_copy["zscore"] = ((s - s.mean()) / s.std()).mean(axis=1)
    return s_copy.values


feature_engineer = FunctionTransformer(create_features)

if __name__ == "__main__":
    with open(f"{BUILD_DIR}/feature_engineer.joblib", "wb") as io:
        dill.dump(feature_engineer, io)

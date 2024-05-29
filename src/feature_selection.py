import dill
from sklearn.feature_selection import SelectKBest
from config import BUILD_DIR


feature_selector = SelectKBest(k="all")

if __name__ == "__main__":
    with open(f"{BUILD_DIR}/feature_selector.joblib", "wb") as io:
        dill.dump(feature_selector, io)

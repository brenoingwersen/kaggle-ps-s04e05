import dill
from sklearn.preprocessing import MinMaxScaler
from config import BUILD_DIR


preprocessor = MinMaxScaler()


if __name__ == "__main__":
    with open(f"{BUILD_DIR}/preprocessor.joblib", "wb") as io:
        dill.dump(preprocessor, io)

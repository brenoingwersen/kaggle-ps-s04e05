import importlib
import dill
from config import BUILD_DIR, MODEL_TYPE


if __name__ == "__main__":
    # Imports the specified model
    model = importlib.import_module(f"models.{MODEL_TYPE}").model
    # Stores it
    with open(f"{BUILD_DIR}/model.joblib", "wb") as io:
        dill.dump(model, io)

import importlib
import dill
from config import BUILD_DIR, MODEL_TYPE

# Imports the specified model
model = importlib.import_module(f"models.{MODEL_TYPE}").model

if __name__ == "__main__":
    # Store the model
    with open(f"{BUILD_DIR}/model.joblib", "wb") as io:
        dill.dump(model, io)

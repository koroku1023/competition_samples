from typing import Any
import pickle


def save_model(model: Any, model_path: str):

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

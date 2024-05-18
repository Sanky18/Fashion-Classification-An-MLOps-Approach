import tensorflow
from tensorflow.keras.models import load_model as keras_load_model

def load_model(path: str):
    """
    Load the Keras Sequential model from the supplied path.
    """
    model = keras_load_model(path)
    return model
import numpy as np

def predict_fashion(model, data_point):
    """
    Predict the fashion using the loaded model and serialized data.
    """
    # Make prediction
    probabilities = model.predict(data_point)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(probabilities[0])
    # Get the confidence score of the predicted class
    confidence_score = np.max(probabilities[0])
    # Get the predicted digit (class)
    predicted_fashion = int(predicted_class_index)
    return predicted_fashion, confidence_score
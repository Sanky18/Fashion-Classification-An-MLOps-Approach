from PIL import Image
import numpy as np
from io import BytesIO

def format_image(image_bytes):
    """
    Resize the uploaded image to 28x28 grayscale, convert it to float32, and normalize the pixel values.
    Returns a 1D array of 784 elements.
    """
    # Convert bytes to image
    img = Image.open(BytesIO(image_bytes))
    # Open and resize the image
    img = img.resize((28, 28)).convert('L')
    # Convert image to numpy array, convert to float32, and normalize
    img = np.array(img)
    img = img.astype('float32') / 255.0
    # Flatten the array to 1D
    img = img.reshape(1, -1)
    return img

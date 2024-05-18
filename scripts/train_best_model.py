import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from plot_keras_history import show_history
from mlflow.models import infer_signature
import os
import numpy as np


Data = np.load('A:\Jan-May 2024\CS5830-Big Data Laboratory\Main Project\Data\\augmented_fashion_mnist.npz')
(X_train, Y_train) = (Data['x_train'], Data['y_train'])
# Load the Fashion MNIST dataset from the Keras library and split it into training and testing sets
(X, Y), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
# Define the number of classes in the dataset (in this case, there are 10 classes for the digits 0-9)
num_classes = 10
# Reshape the input data from 28x28 images to 1D arrays of length 784
x_train = X_train.reshape(60000, 784)  # Reshape training set
x_test = X_test.reshape(10000, 784)     # Reshape testing set
# Convert the data type of the input arrays to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalize the input data by dividing by 255 (maximum pixel value)
x_train /= 255
x_test /= 255
# Print the shape of the training and testing input samples
print(x_train.shape, 'train input samples')
print(x_test.shape, 'test input samples')

# Convert the integer class labels to one-hot encoded vectors using to_categorical function from keras.utils
y_train = keras.utils.to_categorical(Y_train, num_classes)  # Convert training labels
y_test = keras.utils.to_categorical(Y_test, num_classes)    # Convert testing labels
# Print the shape of the training and testing output samples
print(y_train.shape, 'train output samples')
print(y_test.shape, 'test output samples')

learning_rate = 0.001
optimizer = keras.optimizers.Adam

# Define the architecture of the neural network model
model = Sequential([
    Dense(256, activation='sigmoid', input_shape=(784,)),
    Dense(128, activation='sigmoid'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer(learning_rate=learning_rate),
              metrics=['accuracy'])

# Train the model
batch_size = 32
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test))

# Create a directory to save the models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model in the "models" folder
model.save("models/mnist_model.h5")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)
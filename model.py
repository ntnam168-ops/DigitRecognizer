import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

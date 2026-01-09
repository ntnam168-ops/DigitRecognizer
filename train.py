import os
import tensorflow as tf
from model import create_model
from load_mnist import load_images, load_labels

x_train = load_images("D:/AI project/DigitRecognizer/train-images.idx3-ubyte")
y_train = load_labels("D:/AI project/DigitRecognizer/train-labels.idx1-ubyte")
x_test = load_images("D:/AI project/DigitRecognizer/t10k-images.idx3-ubyte")
y_test = load_labels("D:/AI project/DigitRecognizer/t10k-labels.idx1-ubyte")

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = create_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

weights_file = "my_model.weights.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("Loaded existing weights.")

model.fit(x_train, y_train, epochs=5, batch_size=32)

model.save_weights(weights_file)
print("Weights saved to my_model.weights.h5")

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

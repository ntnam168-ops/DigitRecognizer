import os
import time
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from load_mnist import load_images, load_labels

x_test = load_images("D:/AI project/DigitRecognizer/t10k-images.idx3-ubyte")
y_test = load_labels("D:/AI project/DigitRecognizer/t10k-labels.idx1-ubyte")

x_test = x_test / 255.0

x_test = x_test.reshape(-1, 28, 28, 1)

model = create_model()
weights_file = "my_model.weights.h5"

if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("Loaded weights to testing.")

correct = 0
total = len(x_test)

for i in range(total):
    print(i)
    img = x_test[i].reshape(28,28) 

    pred = model.predict(x_test[i].reshape(1,28,28,1))
    predicted_label = pred.argmax()

    print(f"Predicted: {predicted_label}, True: {y_test[i]}")
    if predicted_label == y_test[i]:
        correct += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    ax2.text(0.5, 0.5, f"Predicted: {predicted_label}\nTrue: {y_test[i]}", fontsize=14, ha='center', va='center')
    ax2.axis('off')

    plt.draw()
    plt.pause(1)
    plt.clf() 
    plt.close()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
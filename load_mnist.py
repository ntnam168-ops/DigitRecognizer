import numpy as np

def load_images(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_labels(path):
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

image_train = load_images("D:/AI project/DigitRecognizer/train-images.idx3-ubyte")
answer_train = load_labels("D:/AI project/DigitRecognizer/train-labels.idx1-ubyte")

image_test = load_images("D:/AI project/DigitRecognizer/t10k-images.idx3-ubyte")
answer_test = load_labels("D:/AI project/DigitRecognizer/t10k-labels.idx1-ubyte")
print("Train images:", image_train.shape)
print("Train labels:", answer_train.shape)
print("Test images:", image_test.shape)
print("Test labels:", answer_test.shape)

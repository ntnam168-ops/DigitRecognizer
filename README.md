# Digit Recognizer (MNIST)

This project is a **handwritten digit recognition** system using a **Convolutional Neural Network (CNN)** on the MNIST dataset. It includes training, testing, and visualizing predictions.

---

## Project Structure

- `load_mnist.py`  
  - Loads MNIST dataset from IDX files.  
  - `load_images(path)` → returns `(num_images, 28, 28)`  
  - `load_labels(path)` → returns `(num_labels,)`

- `model.py`  
  - Defines the CNN model with:
    - Conv2D + MaxPooling2D  
    - Flatten  
    - Dense layers  
    - Softmax output (10 classes)  
  - `create_model()` returns a compiled Keras model.

- `train.py`  
  - Loads and normalizes training data, reshapes to `(28,28,1)`.  
  - Creates model via `model.py`, compiles it, loads existing weights if available.  
  - Trains model (`epochs=5`, `batch_size=32`) and saves weights (`my_model.weights.h5`).

- `test.py`  
  - Loads test data, normalizes, reshapes.  
  - Loads trained model and weights.  
  - Displays each test image with predicted and true labels side by side.  
  - Prints **overall test accuracy**.

---

## Dataset

- **MNIST Dataset** (handwritten digits collected from multiple people)  
- Required files (downloaded, not handwritten by you):
  - `train-images.idx3-ubyte` → 60,000 training images  
  - `train-labels.idx1-ubyte` → 60,000 training labels  
  - `t10k-images.idx3-ubyte` → 10,000 test images  
  - `t10k-labels.idx1-ubyte` → 10,000 test labels  

- **Download link:** [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist)  
- Place all files in `D:/AI project/DigitRecognizer/` (or adjust paths in code).  
**Note:** You do not need to write digits yourself.

---

## Requirements

- Python 3.11+  
- Packages:
  ```bash
  pip install tensorflow numpy matplotlib

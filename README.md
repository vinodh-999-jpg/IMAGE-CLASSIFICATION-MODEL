# IMAGE-CLASSIFICATION-MODEL

COMPANY : CODTECH IT SOLUTIONS

NAME : Althi vinodh kumar

INTERN ID : CT04DN428

DOMAIN: MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

##
DESCRIPTION:
Project Title:
Fashion Item Classifier using CNN and Tkinter GUI

Project Overview
This project implements a fashion image classification system that leverages deep learning and a graphical user interface (GUI) to predict clothing item categories. The model is trained on the Fashion MNIST dataset, a widely-used benchmark dataset consisting of grayscale images of clothing and footwear items, making it a modern alternative to the classic MNIST dataset of handwritten digits.

The main objective of this system is to allow users to upload any image of a fashion item (such as a shirt, shoe, or bag), and the model will identify and classify the item into one of the 10 predefined categories. The system is designed to be user-friendly and visually intuitive through a desktop application built using Tkinter, Python’s standard GUI library.

Core Technologies Used
Python: Core programming language for logic and model development.

TensorFlow / Keras: For creating and training the convolutional neural network (CNN).

Fashion MNIST: Built-in dataset containing 70,000 28x28 grayscale images across 10 classes.

PIL (Python Imaging Library): For image processing and display within the GUI.

Tkinter: For building the desktop-based graphical interface.

Functional Components
1. Data Loading and Preprocessing
The system uses the Fashion MNIST dataset, which includes:

60,000 training images

10,000 test images

Each image is a 28x28 grayscale image of a fashion product from one of the following classes:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

Images are normalized (divided by 255.0) to scale pixel values between 0 and 1. They are also reshaped to fit the CNN input shape (28, 28, 1) for grayscale channel compatibility.

2. Model Architecture
The CNN model is designed to efficiently extract and learn spatial hierarchies from the input images. The architecture includes:

Two convolutional layers with ReLU activation

Two max-pooling layers for downsampling

A flatten layer

A dense hidden layer with 64 units

A final softmax layer for multi-class classification

This simple yet powerful architecture achieves strong performance on Fashion MNIST with limited computational requirements.

3. Training and Evaluation
If a model file (fashion_model.h5) does not exist, the system automatically trains the CNN on the training set for 5 epochs and saves the model for future use. The training includes a validation split to monitor overfitting and ensure generalization.

4. Graphical User Interface (GUI)
A major strength of this project is its user-friendly GUI:

The user is greeted with a modern, clean interface.

A “Select Image” button allows the user to choose a fashion image from their computer.

The selected image is resized and displayed.

The model instantly predicts the class, and the result is shown in a readable label below the image.

The GUI ensures accessibility for non-technical users and provides an engaging, real-time classification experience.

Applications and Use Cases
This project demonstrates the practical application of deep learning in the fashion domain. Possible real-world applications include:

Assisting online retail platforms to auto-classify product images.

Developing intelligent wardrobe apps.

Creating visual product recommendation systems.

Building educational tools for learning computer vision.

Conclusion
The Fashion Image Classifier project combines machine learning, image processing, and UI design into a single deployable desktop application. It offers users an interactive way to explore deep learning by allowing them to test predictions using real images. With a well-designed CNN and a clean interface, the system serves both as a learning project and a foundation for more advanced AI applications in fashion tech.
##

# OUTPUT

![Image](https://github.com/user-attachments/assets/80654301-1870-42d0-9fdf-62db4e726e8d)
![Image](https://github.com/user-attachments/assets/ceb57a85-643e-4019-8b1a-07228a82f360)

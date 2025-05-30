import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Model save path
MODEL_PATH = 'fashion_model.h5'

# Image size (as used in Fashion MNIST)
IMG_SIZE = (28, 28)

# Class labels for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Build CNN model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and save model if not already saved
def train_and_save_model():
    print("Training model on Fashion MNIST...")
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize and reshape images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    model = build_model()
    model.fit(train_images, train_labels, epochs=5, validation_split=0.1)
    model.save(MODEL_PATH)
    print("Model trained and saved as fashion_model.h5")

# Predict image from file
def predict_image(model, image_path):
    img = Image.open(image_path).convert('L').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

# Handle image selection and prediction
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        label_result.config(text="Predicting...")
        app.update_idletasks()

        result = predict_image(model, file_path)

        img = Image.open(file_path).resize((200, 200)).convert("RGB")
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        label_result.config(text=f"Prediction: {result}")

# --- MAIN ---

# Train the model if not already saved
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# --- Tkinter UI Setup ---
app = tk.Tk()
app.title("Fashion Item Classifier")
app.geometry("400x520")
app.configure(bg="#ffffff")
app.resizable(False, False)

# Header
header = tk.Label(app, text="Fashion MNIST Classifier", font=("Segoe UI", 18, "bold"), fg="#333", bg="#ffffff")
header.pack(pady=20)

# Image Display Area (blank initially)
image_label = tk.Label(app, bg="#ffffff", borderwidth=2, relief="ridge")
image_label.pack(pady=10)

# Prediction Label
label_result = tk.Label(app, text="Prediction: ", font=("Segoe UI", 14), fg="#222", bg="#ffffff")
label_result.pack(pady=15)

# Button to Select Image
select_btn = tk.Button(
    app,
    text="Select Image",
    command=open_file,
    font=("Segoe UI", 12),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=8
)
select_btn.pack(pady=10)

# Start the application
app.mainloop()
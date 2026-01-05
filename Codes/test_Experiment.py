import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# ---- OPTIONAL: force CPU to avoid CUDA errors ----
# tf.config.set_visible_devices([], "GPU")

Strides = 2

class ConvAutoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 1)),

            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
        ], name="prerncoder")
        
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(16, 16, 128)),

            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu'),

            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ], name="ashicoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def test_on_image(image_path, model_path="/home/ayush/Autonomous_Algo/Codes/best_trained.keras"):

    img = cv2.imread(image_path, 0)

    if img is None:
        print("Error: Could not find image!")
        return

    resized = cv2.resize(img, (256, 256))

    input_data = resized.astype("float32") / 255.0
    input_data = input_data[np.newaxis, ..., np.newaxis]

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ConvAutoencoder": ConvAutoencoder}
    )

    prediction = model.predict(input_data)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(resized, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction.squeeze(), cmap="gray")
    plt.title("Recon")
    plt.axis("off")

    plt.show()


test_on_image("/home/ayush/Autonomous_Algo/Architecture/jero2.png")

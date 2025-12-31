import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Definfing the Architecture. must match with the training script
class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # ENCODER
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
        ], name="encoder")
        
        # DECODER
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(16, 16, 128)),
            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ], name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def test_on_image(image_path, model_path="cat_hq_model.keras"):
    
    # Reading the Image
    img = cv2.imread(image_path, 0) # Read as Grayscale
    
    if img is None:
        print("Error: Could not find image!")
        return

    # Resizingg.....
    original_resized = cv2.resize(img, (128, 128))

    input_data = original_resized.astype("float32") / 255.0
    input_data = input_data[np.newaxis, ..., np.newaxis] 


    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"ConvAutoencoder": ConvAutoencoder}
    )

    # C. Run Prediction
    prediction = model.predict(input_data)
    
    # D. Visualize
    plt.figure(figsize=(10, 5))
    
    # Plot Input
    plt.subplot(1, 2, 1)
    plt.imshow(original_resized, cmap="gray")
    plt.title("Input (128x128)")
    plt.axis("off")
    

    plt.subplot(1, 2, 2)
    # Squeeze removes the extra dimensions (1, 128, 128, 1) -> (128, 128)
    plt.imshow(prediction.squeeze(), cmap="gray") 
    plt.title("Recon")
    plt.axis("off")
    
    plt.show()

test_on_image("/home/ayush/jero.jpg")

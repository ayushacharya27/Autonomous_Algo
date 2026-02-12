from import_datset import split_dataset, preprocess_dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers



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
# Loading Model
model = load_model(
    "/home/ayush/Autonomous_Algo/Training_codes/best_trained.keras",
    custom_objects={"ConvAutoencoder": ConvAutoencoder}
)


# Loading just the Encoder
encoder = model.encoder

# For Training Purposes
folder = "/home/ayush/Autonomous_Algo/DATASET/Flodded_Road"

# Preprocessing the Data
data = preprocess_dataset(folder)

# Extracting the Latents
latents = encoder(data) 

# Rehsaping it, coz it contains a extra dim
latents = tf.reshape(latents, (latents.shape[0], -1))

# Taking the Mean of the Latents
class_signature = tf.reduce_mean(latents, axis=0)

np.save("flood_signature.npy", class_signature.numpy())

print("Saved: flood_signature.npy")

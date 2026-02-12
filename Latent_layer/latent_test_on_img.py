import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import layers



# Autoencoder class

class ConvAutoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(8, 8, 128)),
            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid'),
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)



# load model + encoder
model = load_model(
    "/home/ayush/Autonomous_Algo/Training_codes/best_trained.keras",
    custom_objects={"ConvAutoencoder": ConvAutoencoder}
)

encoder = model.encoder



# Saved Signatures
signatures = {
    "flood": np.load("flood_signature.npy"),
    # add more later
}



# Cosine Similiarity
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)



# Extract latent from one image
def get_latent(img_path):
    img = cv2.imread(img_path, 0)          # grayscale
    img = cv2.resize(img, (256,256))       # must match model size
    img = img.astype("float32") / 255.0
    img = img[..., np.newaxis]             # (128,128,1)
    img = np.expand_dims(img, axis=0)      # (1,128,128,1)

    latent = encoder(img)                  # (1,8,8,128)
    latent = tf.reshape(latent, (latent.shape[0], -1))
    return latent.numpy()[0]



# image_path

img_path = "/home/ayush/Autonomous_Algo/black.png"

latent = get_latent(img_path)

best_class = None
best_score = -1

for cname, sig in signatures.items():
    score = cosine_similarity(latent, sig)
    print(f"{cname} -> {score:.4f}")

    if score > best_score:
        best_score = score
        best_class = cname

print("\nPredicted:", best_class)
print("Score:", best_score)

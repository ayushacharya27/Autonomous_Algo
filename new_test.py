import tensorflow as tf 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize and Reshape
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension (28, 28, 1)
x_train = x_train[..., tf.newaxis]    
x_test  = x_test[..., tf.newaxis]

# ---------------------------------------------------------
# CHANGE IS HERE: Switching to Dense (Linear) Layers
# ---------------------------------------------------------

class SimpleAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder: Flatten image -> Compress to 'latent_dim' (e.g., 64)
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            # FIX: Changed 'relu' to 'sigmoid' to prevent dead neurons (black output)
            layers.Dense(latent_dim, activation='sigmoid'),
        ], name="encoder")
        
        # Decoder: Take 'latent_dim' -> Expand to 784 pixels -> Reshape to image
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ], name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create the model with Latent Dimension = 64
latent_dim = 64
model = SimpleAutoencoder(latent_dim)

# Optimizer
optimizer = tf.keras.optimizers.Adam(1e-3)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------

@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        recon = model(images)
        loss = tf.reduce_mean(tf.square(images - recon)) # MSE Loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Prepare Data Batching
batch_size = 256 
train_ds = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(10000)
    .batch(batch_size)
)

print("Starting Training...")
# Train for 10 Epochs
for epoch in range(250):
    for batch in train_ds:
        loss = train_step(batch)
    print(f"Epoch {epoch+1} | loss = {loss.numpy():.4f}")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

# Pick 10 samples
n = 10
sample = x_test[:n]        # Numpy Array
recon_images = model(sample) # Tensor

plt.figure(figsize=(20, 4))
for i in range(n):
    # Display Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(sample[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Display Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_images[i].numpy().squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
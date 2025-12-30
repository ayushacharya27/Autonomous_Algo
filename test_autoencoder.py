import tensorflow as tf 
from tensorflow.keras import layers

# Loading Dataset MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()


# Why this???
# Since the Images in MNIST are b/w 0-255, we convert it to [0,1] so that the neural network have better chance to predict it coz of its nature
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0


# Why this???
# Shape of x_train: (60000, 28, 28) but, CNN's convolutional layer wants a dimension wether it is one-d(GRAY) or 3-d(RGB), since its RGB we add only one more dim niggers
x_train = x_train[..., tf.newaxis]    
x_test  = x_test[..., tf.newaxis]



def build_encoder():
    # Normal MNIST Layer sizes
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)

    # Size: 7x7x16
    latent = layers.Conv2D(16, 3, padding="same")(x) # We train the model for this

    return tf.keras.Model(inputs, latent, name="encoder_tzu")


def build_decoder():
    # Latents Size
    inputs = layers.Input(shape=(7,7,16)) 

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x) # For Loss Training big cuh

    return tf.keras.Model(inputs, outputs, name="decoder_tzu")


class SimpleAutoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        self.decoder = build_decoder()

    def call(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon



model = SimpleAutoencoder()

# We'll take the Adam Optimizer
optimizer = tf.keras.optimizers.Adam(1e-3)


 
@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        recon = model(images) # The reconstructed Image
        loss = tf.reduce_mean(tf.square(images - recon)) # Comparing with original image

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss # Returning the Loss


# Standard
batch_size = 128

# Important
train_ds = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(10000)
    .batch(batch_size)
)


# Let's train Hurrrrrrayyyyyy!!!
for epoch in range(50):
    for batch in train_ds:
        loss = train_step(batch)

    print(f"Epoch {epoch} | loss = {loss.numpy():.4f}")
    

# But First Let's save this chigga Model
# model.save("mnist_autoencoder") Naah Lets save it different Different

model.encoder.save("mnist_encoder.keras")
model.decoder.save("mnist_decoder.keras")
model.save("mnist_autoencoder.keras")





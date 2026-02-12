import flwr as fl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from import_datset import preprocess_dataset, split_dataset


# Loading Data (The OG Cat Dataset)
all_data = preprocess_dataset("/home/ayush/CAT_00")
x_train, _ = split_dataset(all_data, test_ratio=0.2)

# Well this is Always the Case to Import this
class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(16, 16, 128)),
            layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(1, 3, padding="same", activation="sigmoid"),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))


# Initialising the Model
model = ConvAutoencoder()
model.build((None, 256, 256, 1))

optimizer = tf.keras.optimizers.Adam(1e-3)

# Same Training Method
@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        recon = model(images)
        loss = tf.reduce_mean(tf.square(images - recon))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# Main Client
class AyushClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("Global Model Recieved RAHHHHHH!!!!!")
        model.set_weights(parameters)

        batch_size = 4
        train_ds = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(1000)
            .batch(batch_size)
        )

        for batch in train_ds:
            loss = train_step(batch)

        print(f"Local Training Doneeeee!! | loss={loss.numpy():.4f}")

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        recon = model(x_train[:4])
        loss = tf.reduce_mean(tf.square(x_train[:4] - recon))
        return float(loss.numpy()), len(x_train), {}



SERVER_ADDRESS = "127.0.0.1:8000"  # Currently this is PORT 8000, will change this to NGROK Adress

fl.client.start_numpy_client(
    server_address=SERVER_ADDRESS,
    client=AyushClient(),
)

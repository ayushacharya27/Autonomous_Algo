import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(16, 16, 128)),
            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu'),
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ])

    def call(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

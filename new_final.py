import tensorflow as tf 
from tensorflow.keras import layers

# Loading Dataset MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Converting it to 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Adding Dimension
x_train = x_train[..., tf.newaxis]    
x_test  = x_test[..., tf.newaxis]

class SimpleAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):

        # Initialising the Super Class
        super(SimpleAutoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim


        # Lets Write the Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape = (28,28,1)),
            # We Making it Flat coz Dense Layers take Flat Layers as input
            layers.Flatten(),
            layers.Dense(latent_dim, activation='sigmoid'),
        ], name = "prerncoder")
        
        # Lets Write the Decoder

        self.decoder = tf.keras.Sequential([

            layers.Input(shape = (latent_dim,)),
            layers.Dense(784, activation = 'sigmoid'),
            layers.Reshape((28,28,1))

        ], name = "ashicoder")


    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # I dont know it have to be here
    def get_config(self):
        config = super(SimpleAutoencoder, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


# Lets keep it small
latent_dim = 64

# Initialising my kuchu puchu
model = SimpleAutoencoder(latent_dim)

# Optimizer Nigga Adam
optimizer = tf.keras.optimizers.Adam(1e-3)


# Training Loop For error
@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        recon = model(images)
        loss = tf.reduce_mean(tf.square(images - recon)) # MSE Loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# Batch Size Increased to 256
batch_size = 256 
train_ds = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(10000)
    .batch(batch_size)
)

# Train for 250 Epochs
for epoch in range(250):
    for batch in train_ds:
        loss = train_step(batch)
    print(f"Epoch {epoch+1} | loss = {loss.numpy():.4f}")



model.save("final_autencoder.keras")

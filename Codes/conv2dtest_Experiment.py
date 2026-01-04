import tensorflow as tf 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from import_datset import preprocess_dataset, split_dataset

# Lets First Import the Data
all_data = preprocess_dataset("/home/ayush/CAT_00")
x_train, x_test = split_dataset(all_data, test_ratio=0.2)

# Change this Parameter to increase the Dividing Ratio
Strides = 2



class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Now using 128 by 128 but in future well change it to 256 by 256 for more quality Reconstruction 
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 1)),
            
            # layers.Conv2D(filters, (size_of_convolutional_matrix), strides= how_much_should_it_skip, padding=any_padding_you_want, activation='relu')
            # Block 1: 128 -> 128/2 = 64 (Strides = 2)
            layers.Conv2D(32, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Block 2: 64 -> 32
            layers.Conv2D(64, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Block 3: 32 -> 16
            layers.Conv2D(128, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Final Size 16*16 (Previous)

            # Experiment
            # New Final Size 4*4*512 = 8192
            layers.Conv2D(256, (3,3), strides=Strides, padding='same', activation='relu'),
            layers.Conv2D(512, (3,3), strides=Strides, padding='same', activation='relu'),
     
        ], name="prerncoder")
        
        # Let's Write the Decoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(4, 4, 512)),
            
            # In Conv2DTranspose the Strides adds the extra spaces in the Matrix thus Elongating it

            # Experiment
            layers.Conv2DTranspose(512, (3,3), strides=Strides, padding='same', activation='relu'),
            layers.Conv2DTranspose(256, (3,3), strides=Strides, padding='same', activation='relu'),


            # Un-Block 3: 16 -> 32
            layers.Conv2DTranspose(128, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Un-Block 2: 32 -> 64
            layers.Conv2DTranspose(64, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Un-Block 1: 64 -> 128
            layers.Conv2DTranspose(32, (3,3), strides=Strides, padding='same', activation='relu'),
            
            # Output Layer (Sigmoid for 0-1 pixel values)
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ], name="ashicoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = ConvAutoencoder()
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(images):
    with tf.GradientTape() as tape:
        recon = model(images)
        loss = tf.reduce_mean(tf.square(images - recon)) 
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Batch size can be smaller for larger images
batch_size = 32 
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(batch_size)

# Training Starts 
for epoch in range(100):
    for batch in train_ds:
        loss = train_step(batch)
    print(f"Epoch {epoch+1} | loss = {loss.numpy():.4f}")

model.save("best_trained.keras")

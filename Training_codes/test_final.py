import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Definfing the Architecture. must match with the training script
class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Updated to 256 by 256
        # Now using 128 by 128 but in future well change it to 256 by 256 for more quality Reconstruction 
        self.encoder = tf.keras.Sequential([
            #layers.Input(shape=(128, 128, 1)),

            # New Input
            layers.Input(shape=(256, 256, 1)),

            # Adding one More convolutional layer
            layers.Conv2D(16, (3,3), strides=2, padding='same', activation='relu'),

            # Block 1: 128 -> 64
            layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu'),
            
            # Block 2: 64 -> 32
            layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
            
            # Block 3: 32 -> 16
            layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
            
            # Final Size 16*16
     
        ], name="prerncoder")
        
        # Let's Write the Decoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(16, 16, 128)),
            
            # Un-Block 3: 16 -> 32
            layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu'),
            
            # Un-Block 2: 32 -> 64
            layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu'),
            
            # Un-Block 1: 64 -> 128
            layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu'),

            # Adding one More convolutional layer
            layers.Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu'),
            
            # Output Layer (Sigmoid for 0-1 pixel values)
            layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ], name="ashicoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def test_on_image(image_path, model_path="best_trained.keras"):
    
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

test_on_image("/home/ayush/Autonomous_Algo/Architecture/jero3.jpg")
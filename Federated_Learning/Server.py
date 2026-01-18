import flwr as fl
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

Strides = 2
class ConvAutoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ConvAutoencoder, self).__init__(**kwargs)

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

# Lets Give the Pretrained Model Path and Load it


print("Loading Model Chiggggggaaaaa.....")
model = load_model(
    "/home/ayush/Autonomous_Algo/Training_codes/best_trained.keras",
    custom_objects={"ConvAutoencoder": ConvAutoencoder}
)

# Compiling the Model, WHY?? Because TensorFlow cannot train a model unless it knows how to update weights and how to measure error.
model.compile(
    optimizer="adam",
    loss="mse"
)


# Let's Do Actual Federated Stuff
# OK So what I Decided is to Make a Federated Learning Strategy where i'll Send Global Models to the Client as well as Save the Model after Gettting 5 Rounds of Data from the Clients


class AyushFed(fl.server.strategy.FedAvg):

    # rnd -> Curr Round Number, Results -> Client Updates
    def aggregate_fit(self, rnd, results, failures): 

        # Original Average/Aggregated Function
        aggregated = super().aggregate_fit(rnd, results, failures)

        # This Aggregated Returns 2 Things, (parameters, metrics)
        # parameters -> averaged weights

        if aggregated is not None:
            parameters = aggregated[0] # Still not Numpy
            weights = fl.common.parameters_to_ndarrays(parameters)

            # Finally Setting the Model weights
            model.set_weights(weights)


            # Saving the Model
            model.save("global_autoencoder.keras")
            print(f"Global model updated (round {rnd})")

        
        return aggregated


# FINALLY STARTING THE SERVER

fl.server.start_server(
    server_address = "0.0.0.0:8000",
    strategy = AyushFed(),

    # Save After 5 Rounds
    config=fl.server.ServerConfig(num_rounds=5),
)
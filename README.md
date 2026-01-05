## All Terminologies Used In this
### 1. import_dataset.py
It contains functions:

1. preprocess_dataset(): Helps to clean and preprocess the data
Parameters to change are:

Line 18
```bash
img = cv2.resize(img,(128,128))
```
Change the 128,128 to whatever dimensions you want

2. split_dataset(): Helps to split the dataset into training and test dataset
Parameters to change are:

```bash
def split_dataset(data, test_ratio=0.2):
```
Change the test_ratio to whatever split you want

### 2. conv2dtest_(final).py
It Uses Convolutional layers instead of Dense Layers to extract the feautures from the pictures.

Parameters:
Currently the Latent_dim is set to 16*16, we can make it smaller by adding more convolutional layers into the moedel architecture.

UPDATE (JAN 4, 2026, 11:45 PM): Tried Experimenting by reducing the bottleneck to 4x4x512, Recontruction was bad so left it, Switching back to the old dim with 16x16x128.

NEW PLANNED (JAN 4, 2026, 11:57 PM): First Lets Create the Vector Database to Extract the Data From, and then updating the model to new arch, such as denoising or variational.

UPDATE (JAN 6, 2026, 01:41 AM): Replaced Training Scripts 128x128 by 256x256 for more proper feauture extraction and more trained for more epochs.

### 3. test_(final).py
For Testing the Accuracy of the Model, always use this, make sure to add the **kwargs thingy to the end.

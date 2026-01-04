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

### 2. conv2dtest.py
It Uses Convolutional layers instead of Dense Layers to extract the feautures from the pictures.

Parameters:
Currently the Latent_dim is set to 16*16, we can make it smaller by adding more convolutional layers into the moedel architecture.



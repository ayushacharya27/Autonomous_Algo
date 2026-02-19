## All Terminologies Used In this
### 1. import_dataset.py
It contains functions:

1. **preprocess_dataset()**: Helps to clean and preprocess the data
Parameters to change are:

Line 18
```bash
img = cv2.resize(img,(128,128))
```
Change the 128,128 to whatever dimensions you want

2. **split_dataset()**: Helps to split the dataset into training and test dataset
Parameters to change are:

```bash
def split_dataset(data, test_ratio=0.2):
```
Change the test_ratio to whatever split you want

### 2. conv2dtest_(final).py
It Uses Convolutional layers instead of Dense Layers to extract the feautures from the pictures.

Parameters:
Currently the Latent_dim is set to 16*16, we can make it smaller by adding more convolutional layers into the moedel architecture.

**UPDATE (DEC 28, 2025, 11:45 PM)**: Started Building the AutoEncoder, faced issues with DENSE Layer method, so shifted to Convolutional Layer Strategy.

**UPDATE (JAN 4, 2026, 11:45 PM)**: Tried Experimenting by reducing the bottleneck to 4x4x512, Recontruction was bad so left it, Switching back to the old dim with 16x16x128.

**NEW PLANNED (JAN 4, 2026, 11:57 PM)**: First Lets Create the Vector Database to Extract the Data From, and then updating the model to new arch, such as denoising or variational.

**UPDATE (JAN 6, 2026, 01:41 AM)**: Replaced Training Scripts 128x128 by 256x256 for more proper feauture extraction and more trained for more epochs.

**UPDATE (JAN 9, 2026, 02:25 AM)**: Started Training the Latent Layer and currently working on the Extracting the Latent Layers from the images, Faced an Issue, that i Trained my Encoder on CAT Images, but the Encoder to form the Latent Features, needs to be trained on the Real Road Dataset, Like KITTI and all, have to download it.

**UPDATE (JAN 16, 2026, 01:23 AM)**: Wating for Lab Acess, for training the model (Downloading the Dataset and Training it), so meanwhile started looking at Federated Learning Approaches for Training the Global Model locally on Client Devices.

**UPDATE (JAN 17, 2026, 03:53 AM)**: Tried tensorflow_federated, Fuck this framework, it takes python 3.8 and not compitablw with my current CUDA Version, switiching to something else.

**UPDATE (Jan 17, 2026, 04:21 AM)**: Got a Good Library to Work on, called flowr, framework based on python, used extensively for Federated Learning purpose.

**UPDATE (Jan 19, 2026, 03:05 AM)**: Made Directories,
```bash
mkdir Federated_Learning
```
Started Coding.

**UPDATE (Jan 19, 2026, 04:15 AM)**:  Implemented the Strategy using FLOWR, needs 2 clients, but can override it so no worries. Just sending data to LLM is left and then Hardware Implementaion will be done.

**UPDATE (Feb 13, 2026, 10:07 AM)**: Succesfully demonstrated the Working of Federated Learning Architecture to the CPSD teacher and now to ML Teacher.  

**UPDATE (Feb 19, 2026, 10:52 PM)**: Added 5 New classes to it with dirty road, clean road,.....etc, and finally completed the software part of this project. 


### 3. test_(final).py
For Testing the Accuracy of the Model, always use this, make sure to add the **kwargs thingy to the end.

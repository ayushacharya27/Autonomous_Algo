import numpy as np
import os
import tensorflow as tf
import cv2

def preprocess_dataset(folder_path):
    images = []
    print("Loadinggggggggg.........")

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)

            img = cv2.imread(img_path, 0)

            if img is not None:
                img = cv2.resize(img,(128,128))
                images.append(img)
        else:
            pass


    data = np.array(images).astype("float32") / 255.0
    data = data[..., tf.newaxis]

    return data


def split_dataset(data, test_ratio=0.2):
  
 
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]

    split_index = int(len(data) * (1 - test_ratio)) 


    x_train = data[:split_index]
    x_test  = data[split_index:]

    
    return x_train, x_test







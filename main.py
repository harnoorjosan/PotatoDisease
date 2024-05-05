import tensorflow as tf
import tensorflow.python.keras
#from keras import models, layers
import matplotlib.pyplot as plt
import os

'''
80% data = traning
10% data = validation
10% data = testing
'''

def main():
    BASE_DIR = "PlantVillage"
    IMG_SIZE = 256 # check image proprties
    BATCH_SIZE = 32 # standard convention
    EPOCH = 50 # at end of every epoch, we do validation
    dataset = keras.preprocessing.image_dataset_from_directory(
        directory=BASE_DIR, 
        shuffle=True, 
        image_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE
    )
    class_names = os.listdir(BASE_DIR)
    dataset.
if __name__ == "__main__":
    main()

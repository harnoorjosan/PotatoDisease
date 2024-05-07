import tensorflow as tf
import tensorflow.python.keras
from keras import models, layers
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

'''
80% data = traning
10% data = validation
10% data = testing
'''

def main():
    CLASS_NAMES = ["Early", "Late", "Healthy"]
    MODEL = models.load_model("../models/1.keras")
    img = cv2.imread("03da9931-e514-4cc7-b04a-8f474a133ce5___RS_HL 1830.jpg")
    img_batch = np.expand_dims(img,0)
    prediction = MODEL.predict(img_batch)
    print(prediction)
    #print (confidence)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    print(predicted_class)
    print (confidence)
    '''BASE_DIR = "PlantVillage"
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
    dataset.'''

if __name__ == "__main__":
    main()

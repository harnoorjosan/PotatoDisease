from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import keras

BUCKET_NAME = "harnoor-models"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

model = None

# blob = binary large object = model
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    

def predict(request):
    global model
    if model is not None:
        download_blob(BUCKET_NAME, "models/1.keras", "/tmp/1.keras")
        model = tf.keras.load_model("/tmp/1.keras")
    img = request.files["file"]
    img = np.array(Image.open(img).convert("RGB").resize(256, 256))
    img = img/255 # img pixels should be between 0 and 1


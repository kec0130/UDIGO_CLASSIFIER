import cv2
import numpy as np 
from tensorflow.keras.models import load_model

def inference(model, image_path):
    test = cv2.imread(image_path, cv2.IMREAD_COLOR)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test = cv2.resize(test, (224,224))
    test = test[np.newaxis,:,:,:]
    pred = model.predict(test, batch_size=1)
    return np.argmax(pred)
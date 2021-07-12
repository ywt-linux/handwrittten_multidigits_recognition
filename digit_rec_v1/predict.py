'''
Author: your name
Date: 2021-07-11 15:52:53
LastEditTime: 2021-07-11 20:28:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digit_rec_v1/predict.py
'''
import numpy as np
from keras.models import load_model
from PIL import Image

data = np.load('data/data.npy')
model = load_model('my_model.h5')
result = []
for i in range(5):
    y_pred = model.predict(data[i].reshape(1, 28, 28, 1))
    result.append(np.argmax(y_pred)) # decode one-hot code
print(result)
img = Image.open('after_seperate.png')
img.show()
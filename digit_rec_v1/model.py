'''
Author: your name
Date: 2021-07-11 13:48:41
LastEditTime: 2021-07-11 15:50:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digit_rec_v1/model.py
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten


batch_size=128
num_classes=10
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()


img_row,img_col,channel = 28,28,1

mnist_input_shape = (img_row,img_col,1)

# process data in dataset
train_images = train_images.reshape(train_images.shape[0],img_row,img_col,channel)
test_images = test_images.reshape(test_images.shape[0],img_row,img_col,channel)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images  /= 255
test_images /= 255

# transform to one-hot code
train_labels = keras.utils.to_categorical(train_labels,num_classes)
test_labels = keras.utils.to_categorical(test_labels,num_classes)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),
                    activation="relu",
                    input_shape=mnist_input_shape))
                    # kernalsize = 3*3 
model.add(Conv2D(16,kernel_size=(3,3),
                    activation="relu"
                    ))
model.add(MaxPooling2D(pool_size=(2,2)))
                    
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

model.fit(train_images,
            train_labels,
            batch_size=batch_size,
            epochs=2,
            verbose=1,
            validation_data=(test_images,test_labels),
            shuffle=True
            )

score = model.evaluate(test_images,test_labels,verbose=1)

print('test loss:',score[0])
print('test accuracy:',score[1])
model.save('my_model.h5')
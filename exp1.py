import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.02,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train=train_datagen.flow_from_directory("archive/chest_xray/chest_xray/train",target_size=(150,150),batch_size=32,class_mode='binary')
test=test_datagen.flow_from_directory("archive/chest_xray/chest_xray/test",target_size=(150,150),batch_size=32,class_mode='binary')
#32 filters of matrix size 3*3
#input_shape => 150*150 image with 3 colors RGB
model=Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),tf.keras.layers.MaxPooling2D(2,2),#converto to 2D
                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation='relu'),
                  tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train,validation_data=test,epochs=5)
model.summary()
model.save("pneumonia_model.h5")
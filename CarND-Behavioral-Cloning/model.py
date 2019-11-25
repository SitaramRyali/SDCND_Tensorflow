import csv
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import Callback,ModelCheckpoint


#import keras.layers.pooling.MaxPooling2D as MaxPooling2D
# example of loading the vgg16 model
#from keras.applications.vgg16 import VGG16,decode_predictions

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
##load the data into images list
images =[]
measurements = []
c = 0
for line in lines:
    if c==0:
        c += 1
        continue
        

    im_path = 'data/'
    for i in range(3):
        source_path = line[i]
        filename = source_path.split(' ')[-1]
        current_path = im_path+filename
        #print(current_path)
        image= cv2.imread(current_path)
        #newimg = cv2.resize(image,(224,224))

        images.append(image) 
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+0.2)
    measurements.append(measurement-0.2)
        
print(len(images))
print(len(measurements))
##augmenting the images
augmented_images,augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
        
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
y_train = y_train.reshape(-1,1)
#x_train = (x_train-255.0) - 0.5
    



print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape =(160,320,3)))
model.add(Cropping2D(cropping =((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2, input_shape=(64,3,3)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
#model.summary()
#compile the defined model
#early stopping function

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.0200, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint('model2.h5', monitor='val_loss', save_best_only=True, verbose=0),
]
model.compile(loss = 'mse',optimizer = 'adam')

model.fit(x_train, y_train,validation_split = 0.010, epochs=10,
      shuffle=True, verbose=1,callbacks=callbacks)
#model.fit(x_train,y_train,validation_split = 0.2, shuffle = True,epochs =10)
model.save('model.h5')



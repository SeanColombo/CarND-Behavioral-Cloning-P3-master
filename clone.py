import csv
import cv2
import numpy as np

####
# NOTES: I didn't do the flipping images for data augmentation because I drove my car around the track twice
# in each direction, so there shouldn't be a directional bias.
#
# TODO: Consider doing the trick with the multiple cameras & tweaked steering angles (Lesson 12-12).
# TODO: Do some "recovery" sections (when heading around the track in either direction)... only record the correction, not the initial driving to the edge.
# TODO: Use track 2?  This will make it more general-purpose but is probably only good for after it can drive track 1.
####


#INPUT_DIR = "data" # DATA THAT COMES WITH THE PROJECT
#INPUT_DIR = "seandata" # Me, driving around, one lap.
INPUT_DIR = "seandata_long" # Me, driving twice in each direction. This removes the bias for turning left.
# Current Data in seandata_long:
#   Lap
#   Backwards Lap
#   Lap
#   Backwards Lap


lines = []
with open("./"+INPUT_DIR+"/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
  
images = []
measurements = []
for line in lines:
    if(line[0] == "center"):
        continue # skip the header line
    else:
        source_path = line[0]
        # To handle all operating systems, we stack the operations to pull off JUST the filename
        filename = source_path.split('/')[-1] # non-windows
        filename = source_path.split("\\")[-1] # windows
        current_path = "./"+INPUT_DIR+"/IMG/" + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Cropping2D
#from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# == PREPROCESSING ==
# Normalize the pixels to be -0.5 to 0.5.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Crop out the top and bottom  pixels which are sky or car-hood.
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

# == MODEL ==
# Roughly, a LeNet architecture. This has really low loss but still doesn't drive well.
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# The NVIdia architecture
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
# Default data:
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

output_file = 'model_'+INPUT_DIR+'.h5'
model.save(output_file)
print("Model saved to file: ",output_file)

import csv
import cv2
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,Cropping2D
#from keras.layers.convolutional import Convolution2D # seems to have entirely different method signatures
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

####
# NOTES: I didn't do the flipping images for data augmentation because I drove my car around the track twice
# in each direction, so there shouldn't be a directional bias.
#
# USAGE:
# Train on the long dataset using (with seandata_long as INPUT_DIR in this script):
#   python model.py
# Run the simulator:
#   Open windows_sim.exe (click Autonomous mode & wait for car to get dropped on track)
#   python drive.py model.h5
# To create a video of its performance:
#   python drive.py model.h5 run1
#   python video.py run1 --fps 48
#
#
# TODO: Generators like in Lesson 12-17 so that it doesn't have to try to load all of the images into memory at once.
# TODO: When my graphics card gets here, make sure that's actually being used.
# TODO: Do a "recovery" lap in each direction (when heading around the track in either direction)... only record the correction, not the initial driving to the edge.
# TODO: Add dropout layers to prevent overfitting (it's in the rubric!)
# TODO: Read tips in the PDF.
#
#
# IF WE GET STUCK:
# TODO: Consider doing the trick with the multiple cameras & tweaked steering angles (Lesson 12-12).
# TODO: Use track 2?  This will make it more general-purpose but is probably only good for after it can drive track 1.
# TODO: Check MSE of whole model. If high, it is underfitting. If low on training but high on validation, then it is overfitting. Collecting more data can help against overfitting.
#       If low MSE but car is going off of track, it probably needs more data in situations where it goes off the track.
####


#INPUT_DIR = "data" # DATA THAT COMES WITH THE PROJECT
#INPUT_DIR = "seandata" # Me, driving around, one lap.
INPUT_DIR = "seandata_long" # Me, driving twice in each direction. This removes the bias for turning left.
# Current Data in seandata_long:
#   Lap
#   Backwards Lap
#   Lap
#   Backwards Lap
#   Trained another forward pass at just that spot near the dirt-road tire-trap area after the bridge.


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
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, verbose=1)

#output_file = 'model_'+INPUT_DIR+'.h5'
output_file = "model.h5" # the project defines this as the required convention
model.save(output_file)
print("Model saved to file: ",output_file)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
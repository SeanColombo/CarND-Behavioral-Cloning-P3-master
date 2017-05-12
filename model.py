import csv
import cv2
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Lambda,Convolution2D,Cropping2D
#from keras.layers.convolutional import Convolution2D # seems to have entirely different method signatures
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle
import matplotlib.pyplot as plt

####
# NOTES:
#   - I didn't do the flipping images for data augmentation because I drove my car around the track twice
#     in each direction, so there shouldn't be a directional bias.
#   - Upgraded to a GTX 1060 (was processing on CPU before). Running 1 epoch went from 44 seconds to 7 seconds.
#   - Added Dropout between some of the fully-connected layers to prevent overfitting (as per ruberic). Didn't really
#     see overfitting yet, but the rubric mandated it, so it's clear it would happen eventually.
#       Results: the driving became noticeably smoother (less correcting back & forth) but it also lost the ability it had
#       just learned, to avoid the dirt-trap after the bridge.  Will add more training data.
#   - Added Generators to allow my dataset to grow well beyond what it is now. This appears to have impacted performance (9s to train instead of 7s) but
#     that could be partly related to how the timing counts... I don't think the counting previously included the time that it took to load the images
#     into memory (which took several seconds for even 4 laps of images).  I experimented with batch size and even though my memory was able to handle
#     batch sizes up to 512 without a problem, 64 seemed to be a good batch-size for speed (512 was consistently about one second slower... there must have
#     been some I/O side-effect or something).
#   - Trained more in the dirt-trap area.
#   - Since failure occurs so late in the lap (the dirt trap is a like a minute in, at 9mph) I modified drive.py on line 47 to be "set_speed = 30" instead of 9.
#     This doesn't seem to have created any additional failure. It doesn't drive QUITE as well, but it fails near the same spot... actually it gets past the dirt
#     trap by a very slight amount before exiting the track.  I'll probably continue the project at this speed. It is also the speed at which most of my data
#     has been trained. FWIW: I tried setting it higher too, and that doesn't seem to be an option. The controller sets a "governor" at around 30mph.
#   - Loss & val loss are both around 0.02X at this point, but the car doesn't make it around the track yet.
#   - Got more data for training the recovery laps.  Tried 5 epochs. 2 was almost as good (training loss was better on 5, but val loss was same).
#   - After adding recovery laps & 5 epochs, the car now goes around the track indefinitely!  It touches the red/white checks at some points (after dirt trap & during the next turn).
#     Interestingly, adding the recovery training actually made most of the driving more wobbly instead of cruising smoothly in the center of the lane, but the overall performance is way more effective.
#   - There are exactly two challenging spots left. I'm going to train more recovery info for those types of cases, BEFORE getting into the problem to help avoid it.
#   - NOTE: All of my recovery training appears to increase Loss, but that's fairly expected. The smooth driving was done at the beginning.
#   - After the additional recovery training (catching cases I didn't get in my Recovery Laps) I realized that epochs were still making progress at more epochs, however the model started to over-fit.
#     I experimented with adding an additional dropout layer between the last two fully-connected layers (which seemed dangerous because there is only one output) and with increasing the number of epochs to 7 or 10. This
#     was able to slightly improve the loss numbers but made my car drive worse (and even get stuck in one spot). As long as the training and val loss look reasonable, I think I'll stick with judging based on
#     performance on the track.
#
# 
# Current Data in seandata_long:
#   Lap
#   Backwards Lap
#   Lap
#   Backwards Lap
#   Trained a bit near dirt-road tire-trap forward
#   Trained a bit near dirt-road tire-trap forward (9-10 mph)
#   Trained a bit near dirt-road tire-trap backwards (9-10 mph)
#   Trained recovery lap forward
#   Trained recovery lap backwards
#   Trained recovery near two bad spots (block after dirt trap & end of the next right turn) forward. About 3 approaches to each area.
#
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
#
# TODO: Read tips in the PDF.
#   - Tip: "Resize the input image. I was able to size the image down by 2, reducing the number of pixels by 4. This really helped speed up model training and did not seem to impact the accuracy."
#   - Tip: It says he needed 40k samples. At 6,400 I'm doing laps (but touching the red/white checks in two spots).
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


# == GENERATOR CODE ==
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                # To handle all operating systems, we stack the operations to pull off JUST the filename
                filename = source_path.split('/')[-1] # non-windows
                filename = source_path.split("\\")[-1] # windows
                current_path = "./"+INPUT_DIR+"/IMG/" + filename
                center_image = cv2.imread(current_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Read the CSV file which contains info about all of the training images.
lines = []
with open("./"+INPUT_DIR+"/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# This is the straightforward method of processing the images without using generators.
# images = []
# measurements = []
# for line in lines:
    # if(line[0] == "center"):
        # continue # skip the header line
    # else:
        # source_path = line[0]
        ##To handle all operating systems, we stack the operations to pull off JUST the filename
        # filename = source_path.split('/')[-1] # non-windows
        # filename = source_path.split("\\")[-1] # windows
        # current_path = "./"+INPUT_DIR+"/IMG/" + filename
        # image = cv2.imread(current_path)
        # images.append(image)
        # measurement = float(line[3])
        # measurements.append(measurement)
# X_train = np.array(images)
# y_train = np.array(measurements)
        
# This is the method of processing the data using generators.
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

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

DROP_RATE = 0.3 # how much to drop in the dropout later, from 0 (don't drop) to 1 (drop everything).

# The NVIdia architecture
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(DROP_RATE))
model.add(Dense(50))
model.add(Dropout(DROP_RATE))
model.add(Dense(10))
#model.add(Dropout(DROP_RATE))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
# Default data:
# The method when not using generators:
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, verbose=1)
# The method when using generators:
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples), \
                                     nb_epoch=5, verbose=1)

#output_file = 'model_'+INPUT_DIR+'.h5'
output_file = "model.h5" # the project defines this as the required output filename convention
model.save(output_file)
print("Model saved to file: ",output_file)


# print(history_object.history.keys())
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

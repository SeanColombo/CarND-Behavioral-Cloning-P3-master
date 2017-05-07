import csv
import cv2
import numpy as np

#INPUT_DIR = "data" # DATA THAT COMES WITH THE PROJECT
INPUT_DIR = "seandata" # Me, driving around

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
from keras.layers import Flatten,Dense
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# Default data:
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
# My data seems to learn for longer.
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

output_file = 'model_'+INPUT_DIR+'.h5'
model.save(output_file)
print("Model saved to file: ",output_file)

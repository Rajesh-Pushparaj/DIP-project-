import numpy as np
import pandas as pd
from pendulum_class import *
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import os

# initial state of the pendulum
x0 = np.array([0.5,np.pi,np.pi,1.0,0.0,0.0]) # initial state of the pendulum

# Load the pendulum
pendulum = pendulum_simulator(x0)

#path
root = os.getcwd()

#Load models 

Model1_path_json  = root + '\Imitation_model.json'
Model1_path_h5 = root + '\Imitation_model.h5'

# load json and create model
json_file = open(Model1_path_json ,'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights(Model1_path_h5)
print("Loaded model 1 from disk")



Model2_path_json  = root + '\Imitation_model.json' #Change the model name as need
Model2_path_h5 = root + '\Imitation_model.h5'

# load json and create model
json_file = open(Model2_path_json ,'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights into new model
model2.load_weights(Model2_path_h5)
print("Loaded model 2 from disk")

#  simulate pendulum for random inputs
for _ in range(250):
	x_in = pd.DataFrame(x0)
	# compute optimal control input via MPC
	if ((x0[1] <= np.pi/8)and(x0[2] <= np.pi/8)) and ((x0[1] >= -np.pi/8)and(x0[2] >= -np.pi/8)):
		u0 = model2.predict(x_in.T)
	else:
		u0 = model1.predict(x_in.T)
	# Simulate pendulum
	x0 = pendulum.simulate(u0)
	
# Generate a gif containing the results
pendulum.export_gif()

# Export data for learning
#pendulum.export_data('filename')

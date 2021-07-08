import numpy as np
import pandas as pd
from pendulum_class import *
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import os

# initial state of the pendulum
x0 = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # initial state of the pendulum

# Load the pendulum
pendulum = pendulum_simulator(x0)

# load mpc controller and set initial state and guess
# mpc = template_mpc()
# mpc.x0 = x0
# mpc.set_initial_guess()


# load json and create model
json_file = open(r"C:\Users\HP\Documents\GitHub\DIP-project-\src_RL\RL_json\pendulum_target_actor.json")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(r"C:\Users\HP\Documents\GitHub\DIP-project-\src_RL\RL_json\pendulum_target_actor.h5")
print("Loaded model from disk")

# # load json and create model
# json_file = open(r"C:\Users\HP\Documents\GitHub\DIP-project-\src_RL\RL_json\pendulum_actor.json")
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(r"C:\Users\HP\Documents\GitHub\DIP-project-\src_RL\RL_json\pendulum_actor.h5")
# print("Loaded model from disk")

#  simulate pendulum for random inputs
for _ in range(250):
	x_in = pd.DataFrame(x0)
	# compute optimal control input via MPC
	u0 = model.predict(x_in.T)
	# Simulate pendulum
	x0 = pendulum.simulate(u0)
	
# Generate a gif containing the results
pendulum.export_gif()

# Export data for learning
#pendulum.export_data('filename')

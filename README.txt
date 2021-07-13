# Readme - MLME project - Deep learning-based control

Library dependency
1) Install the following libraries - Pandas, Keras, Tensorflow, numpy, sklearn, os, matplotlib, gym, Imagemagick 

Imitation Learning
1) Dataset required for Imitiation Learning can be found inside the folder named 'Dataset'.
2) Imitation model was trained and validated using the file named 'imitation_learning_model.ipynb'.
3)a) For verifying the imitation model, run the file named 'Im_control_2Mod.py' (2 models- switchnig for swing up & balancing)
  b) For verifying the imitation model, run the file named 'Imitation_control.py' (1 model)
4) After running the file GIF gets generated and the result can be verified.

Reinforcement Leanring (go to the dir -..\DIP-project-\src_RL\RL_json
1) Register the gym environment file 'DIP_env.py' in the root path of gym.
2) Run the file 'ddpg_trial.py' to execute the DDPG algorithm.
3) The Actor model (.h5) is saved in the current path of RL
4) For verifying the RL actor model, run the file named 'rl_1.py'
5) After running the file GIF gets generated and the result can be verified. 
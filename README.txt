Table of Contents

DroneMA.py - The environment that uses the dense reward function
DroneMA_Sparse - The environment that uses the sparse reward function
DroneMALoad - A file used to apply a wrapper to DroneMA_Sparse when it is loaded by DroneMATrainer.
DroneMALoad - A file used to apply a wrapper to DroneMA when it is loaded by DroneMATrainer.
DroneMATrainer - The Code used to train the drones on either environment

Both environments are capable of single agent and multi agent training, but the amount of agents in the environment need to be changed manually.
Maps can be changed by editing the map variable in both init and reinit.
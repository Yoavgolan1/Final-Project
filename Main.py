import numpy as np
import trajectories as trj
from backprop import Backprop
# Create training set

n_trajectories = 100
trajectory_length = 100
split_ratio = 0.99

inputs_length = 2*np.round(split_ratio * trajectory_length).astype(int)
outputs_length = 2*trajectory_length - inputs_length
inputs = np.zeros((n_trajectories, inputs_length))
outputs = np.zeros((n_trajectories, outputs_length))
print(outputs.shape)
for ii in range(n_trajectories):
    new_traj = trj.create_trajectory(n_timesteps=trajectory_length, g=0.0981, initial_speed_interval=(0, 1))
    new_input, new_output = trj.split_trajectory(new_traj, split_ratio)
    inputs[ii, :] = new_input
    outputs[ii, :] = new_output

bp = Backprop(inputs_length, 50, outputs_length)
bp.train(inputs, outputs)

print(outputs[0, :])
print(bp.test(inputs)[0])
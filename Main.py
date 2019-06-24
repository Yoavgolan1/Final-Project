import numpy as np
import trajectories as trj
from backprop import Backprop
import matplotlib.pyplot as plt

# Create training set

n_trajectories = 3
trajectory_length = 100
split_ratio = 0.9

inputs_length = 2*np.round(split_ratio * trajectory_length).astype(int)
outputs_length = 2*trajectory_length - inputs_length
inputs = np.zeros((n_trajectories, inputs_length))
outputs = np.zeros((n_trajectories, outputs_length))
#print(outputs.shape)
fig = plt.figure()

for ii in range(n_trajectories):
    new_traj = trj.create_trajectory(n_timesteps=trajectory_length, duration = 14, initial_speed_interval=(150, 150), initial_angle_interval=(44, 45), noise=1)
    new_input, new_output = trj.split_trajectory(new_traj, split_ratio)
    inputs[ii, :] = new_input
    outputs[ii, :] = new_output
    # trj.plot(new_traj[0], new_traj[1])

    input_len = np.round(split_ratio * trajectory_length).astype(int)
    outputs_len = trajectory_length - input_len
    plt.plot(inputs[0,:input_len], inputs[0,input_len:], 'o-r')
    plt.plot(outputs[0, :outputs_len], outputs[0, outputs_len:], 'o-b')

# plt.show()
inputs = np.asarray(inputs)
outputs = np.asarray(outputs).T
# bp = Backprop(inputs_length, 50, outputs_length)
bp = Backprop(input_len*2, outputs_len*2, 2)
bp.train(inputs, outputs)
#
# print(outputs[0, :])
# print(bp.test(inputs)[0])


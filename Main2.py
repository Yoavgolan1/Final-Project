import numpy as np
import trajectories as trj
from backprop import Backprop
import matplotlib.pyplot as plt

# Create training set
n_trajectories = 10
trajectory_length = 100
# split_ratio = 0.9

inputs_length = 2*np.round(trajectory_length).astype(int)
outputs_length = 2*trajectory_length - inputs_length
inputs = np.zeros((n_trajectories, inputs_length))
outputs = np.zeros((n_trajectories, outputs_length))
fig = plt.figure()

Itraj = np.zeros((n_trajectories, trajectory_length))
Ttraj = np.zeros((n_trajectories, 1))
for ii in range(n_trajectories):
    Itraj[ii], Ttraj[ii] = trj.create_trajectory(n_timesteps=trajectory_length, duration = 14, initial_speed_interval=(150, 150), initial_angle_interval=(44, 45), noise=1)

    marker = '--r'
    if Ttraj[ii] == True:
        marker = '--g'
    plt.plot(Itraj[ii,:trajectory_length/2], Itraj[ii,trajectory_length/2:], marker, alpha = 0.1)

plt.show()

bp = Backprop(n = trajectory_length, m = 1, h = 2)
bp.train(Itraj, Ttraj)

Itest, Ttest = trj.create_trajectory(n_timesteps = trajectory_length, duration=14, initial_speed_interval = (150, 150), initial_angle_interval = (44, 45), noise = 1)
O = bp.test(Itest)

fig = plt.figure()
line_color = 'r'
if O == Ttest:
    line_color = 'g'
line_style = '--'
if O == Ttest:
    line_style = '-'
plt.plot(Itraj[ii, :trajectory_length/2], Itraj[ii, trajectory_length/2:], linestyle = line_style, color = line_color, alpha = 0.1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test result')
plt.show()

#
# print(outputs[0, :])
# print(bp.test(inputs)[0])


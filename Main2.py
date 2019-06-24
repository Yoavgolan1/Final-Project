import numpy as np
import trajectories as trj
from backprop_yo import Backprop
import matplotlib.pyplot as plt

# Create training set
n_trajectories = 1000
trajectory_length = 100
# split_ratio = 0.9

inputs_length = 2*np.round(trajectory_length).astype(int)
outputs_length = 2*trajectory_length - inputs_length
inputs = np.zeros((n_trajectories, inputs_length))
outputs = np.zeros((n_trajectories, outputs_length))
fig = plt.figure()

Itraj = np.zeros((n_trajectories, trajectory_length*2))
Ttraj = np.zeros((n_trajectories, 1))
basket = [(40,10), (45,10)]
plt.plot(np.asarray(basket)[:,0], np.asarray(basket)[:,1], 'rd-')
for ii in range(n_trajectories):
    # Itraj[ii, :], Ttraj[ii]
    Itraj[ii, :], Ttraj[ii] = trj.create_trajectory(n_timesteps=trajectory_length, duration = 5,
                                                 initial_speed_interval=(15, 45), initial_angle_interval=(40, 80),
                                                 noise = 0.1, basket = basket)
    # Itraj[ii, :] = np.asarray(Itemp)
    # Ttraj[ii, 0] = Ttemp
    # marker = '--r' if Ttraj[ii] == True else marker = '--g'
    marker = '--r'
    if Ttraj[ii] == True:
        marker = '--g'

    plt.plot(Itraj[ii,:trajectory_length], Itraj[ii,trajectory_length:], marker, alpha = 0.1)
# plt.show()

bp = Backprop(n = trajectory_length*2, m = 1, h = 8)
bp.train(Itraj, Ttraj, niter = 1000)

## Test
fig = plt.figure()
for i in range (20):
    Itest, Ttest = trj.create_trajectory(n_timesteps = trajectory_length, duration =5,
                                     initial_speed_interval = (20, 25), initial_angle_interval = (50, 62), noise = 0.1, basket = basket)
    O = bp.test(Itest)
    plt.plot(np.asarray(basket)[:,0], np.asarray(basket)[:,1], 'rd-')
    # line_color = 'g' if O == Ttest else line_color = 'r'
    # line_style = '-' if O == Ttest else line_style = '--'
    line_color = 'r'
    print(O)
    if O > 0.5:
        line_color = 'g'
    # plt.plot(Itest[0, :trajectory_length], Itest[0, trajectory_length:], linestyle = '-', color = line_color, alpha = 0.1)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test result')
plt.show()

#
# print(outputs[0, :])
# print(bp.test(inputs)[0])


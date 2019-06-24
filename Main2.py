import numpy as np
import trajectories as trj
from backprop_old import Backprop
import matplotlib.pyplot as plt

# Create training set
n_trajectories = 5000
trajectory_length = 10
# split_ratio = 0.9

inputs_length = 2*np.round(trajectory_length).astype(int)
outputs_length = 2*trajectory_length - inputs_length
inputs = np.zeros((n_trajectories, inputs_length))
outputs = np.zeros((n_trajectories, outputs_length))
fig = plt.figure()

Itraj = np.zeros((n_trajectories, trajectory_length*2))
Ttraj = np.zeros((n_trajectories, 1))

basket = [(100, 10), (140, 30)]


plt.plot(np.asarray(basket)[:, 0], np.asarray(basket)[:, 1], 'rd-')
for ii in range(n_trajectories):
    # Itraj[ii, :], Ttraj[ii]
    Itraj[ii, :], Ttraj[ii] = trj.create_trajectory(n_timesteps=trajectory_length, duration = 5,
                                                 initial_speed_interval=(15, 45), initial_angle_interval=(40, 80),
                                                 noise =0.1, basket = basket)
    # Itraj[ii, :] = np.asarray(Itemp)
    # Ttraj[ii, 0] = Ttemp
    # marker = '--r' if Ttraj[ii] == True else marker = '--g'
    marker = '--r'
    if Ttraj[ii] == True:
        marker = '--g'

    plt.plot(Itraj[ii, :trajectory_length], Itraj[ii,trajectory_length:], marker, alpha = 0.1)
# plt.show()

bp = Backprop(n = trajectory_length*2, m = 1, h = 50)
#bp.train(Itraj, Ttraj, iterations=1000, eta=0.3, mu=0.05, lambada=0.95)
bp.load("weights")
bp.save("weights")
## Test
fig = plt.figure()

OO = np.zeros(200)
for i in range(200):
    Itest, Ttest = trj.create_trajectory(n_timesteps = trajectory_length, duration =5,
                                     initial_speed_interval = (15, 45), initial_angle_interval = (40, 80), noise = 0.1, basket = basket)
    OO[i] = bp.test(Itest)

plt.hist(OO)
plt.figure()

for i in range(200):
    Itest, Ttest = trj.create_trajectory(n_timesteps = trajectory_length, duration =5,
                                     initial_speed_interval = (15, 45), initial_angle_interval = (40, 80), noise = 0.1, basket = basket)
    O = bp.test(Itest)
    plt.plot(np.asarray(basket)[:,0], np.asarray(basket)[:,1], 'rd-')
    # line_color = 'g' if O == Ttest else line_color = 'r'
    # line_style = '-' if O == Ttest else line_style = '--'
    line_color = 'r'
    #print(O)
    if O > 0.30:
        line_color = 'g'
    line_style = '--'
    if Ttest ==1:
        line_style = '-'
    plt.plot(Itest[0, :trajectory_length], Itest[0, trajectory_length:], linestyle = line_style, color = line_color, alpha = 0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test result')
plt.show()

#
# print(outputs[0, :])
# print(bp.test(inputs)[0])


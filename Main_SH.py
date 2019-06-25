import numpy as np
import trajectories2 as trj
from backprop_old import Backprop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create training set
n_trajectories = 10
dt = 0.01
time_interval = (0,5)
n_time_steps = int((time_interval[1] - time_interval[0]) / dt + 1)

# inputs_length = 2*np.round(trajectory_length).astype(int)
# outputs_length = 2*trajectory_length - inputs_length
# inputs = np.zeros((n_trajectories, inputs_length))
# outputs = np.zeros((n_trajectories, outputs_length))
fig = plt.figure()
ax = fig.gca(projection='3d')
Itraj = np.zeros((n_trajectories, n_time_steps*3))
Ttraj = np.zeros((n_trajectories, 3))

for ii in range(n_trajectories):
    Itraj[ii, :], Ttraj[ii] = trj.create_trajectory(dt = dt, time_interval = time_interval,
                                                    initial_speed_interval=(25, 45), initial_angleAZ_interval=(1, 89),
                                                    initial_angleAL_interval=(50, 80), noise =0.3,)

    x_pos = Itraj[ii, 0:n_time_steps]
    y_pos = Itraj[ii, n_time_steps:n_time_steps*2]
    z_pos = Itraj[ii, n_time_steps*2:]
    color = np.random.rand(3)
    ax.plot(x_pos, y_pos, z_pos, '-', color = color, alpha = 1)
    ax.plot([Ttraj[ii,0]], [Ttraj[ii,1]], [Ttraj[ii,2]], 'x', color = color)
plt.xlabel('X')
plt.ylabel('Y')
plt.zlabel('Z')
plt.show()
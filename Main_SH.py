import numpy as np
import trajectories2 as trj
from backprop_old import Backprop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create training set
n_trajectories = 1000
dt = 0.1
time_interval = (0,90)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 60)
initial_angleAL_interval = (45, 50)
n_time_steps = int((time_interval[1] - time_interval[0]) / dt + 1)

fig = plt.figure()
ax = fig.gca(projection = '3d')
Itraj = np.zeros((n_trajectories, n_time_steps*3))
Ttraj = np.zeros((n_trajectories, 3))
RMS = np.zeros(n_trajectories)
ePos = np.zeros(n_trajectories)
for ii in range(n_trajectories):
    Itraj[ii, :], Ttraj[ii, :] = trj.create_trajectory(dt = dt, time_interval = time_interval,
                                                       initial_speed_interval = initial_speed_interval,
                                                       initial_angleAZ_interval = initial_angleAZ_interval,
                                                       initial_angleAL_interval = initial_angleAL_interval,
                                                       noise = 20)
    target_estimated = trj.solver(dt, Itraj[ii, :])
    RMS[ii] = trj.RMS(target_estimated, Ttraj[ii, :])
    ePos[ii] = trj.error(target_estimated, Ttraj[ii, :])
    # print("RMS = {:.3f}".format(RMS[ii]))

    # Plot each trajectory
    x_pos = Itraj[ii, 0:n_time_steps]
    y_pos = Itraj[ii, n_time_steps:n_time_steps*2]
    z_pos = Itraj[ii, n_time_steps*2:]
    color = np.random.rand(3)
    ax.plot(x_pos, y_pos, z_pos, '-', color = color, alpha = 1)
    ax.plot([Ttraj[ii,0]], [Ttraj[ii,1]], [Ttraj[ii,2]], 'x', color = color)
    ax.plot([target_estimated[0]], [target_estimated[1]], [target_estimated[2]], '.', color=color)

print("Mean RMS = {:.3f}".format(np.mean(RMS)))
print("Mean error pos = {:.3f}".format(np.mean(ePos)))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.show()

plt.figure()
t = np.arange(time_interval[0], time_interval[1]+dt, dt)
plt.subplot(2, 1, 1)
x_pos = Itraj[ii, 0:n_time_steps]
y_pos = Itraj[ii, n_time_steps:n_time_steps*2]
z_pos = Itraj[ii, n_time_steps*2:]
plt.plot(t, x_pos, 'r')
plt.plot(t, y_pos, 'b')
plt.plot(t, z_pos, 'k')
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')

plt.subplot(2, 1, 2)
n_start = 0
n_end = int(0.5/dt+1)
plt.plot(t[n_start:n_end], x_pos[n_start:n_end], '-+r')
plt.plot(t[n_start:n_end], y_pos[n_start:n_end], '-+b')
plt.plot(t[n_start:n_end], z_pos[n_start:n_end], '-+k')
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')

plt.figure()
plt.hist(ePos)
plt.show()
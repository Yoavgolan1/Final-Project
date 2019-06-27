import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D  # This looks unused but it is used


# Create training set
def create_dataSet(n_trajectories=100, dt=0.1, time_interval=(0, 10), initial_speed_interval=(0, 10),
                      initial_angleAZ_interval=(15, 165), initial_angleAL_interval=(15, 165),
                      noise=20, g=9.81, plot=False):
    # Create data set of different trajectories using the "create_trajectory" function.
    # That data set is saved to a CSV file and then being used for learning
    n_time_steps = int((time_interval[1] - time_interval[0]) / dt + 1)
    Itraj = np.zeros((n_time_steps, 3 * n_trajectories))
    Ttraj = np.zeros((3, 3 * n_trajectories))
    RMS = np.zeros(n_trajectories)
    ePos = np.zeros(n_trajectories)
    for ii in range(n_trajectories):
        Itraj[:, (ii * 3):(ii * 3 + 3)], Ttraj[0, (ii * 3):(ii * 3 + 3)] =\
            create_trajectory(dt=dt, time_interval=time_interval,
                              initial_speed_interval=initial_speed_interval,
                              initial_angleAZ_interval=initial_angleAZ_interval,
                              initial_angleAL_interval=initial_angleAL_interval, noise=noise, g=g)

        Ttraj[1, (ii * 3):(ii * 3 + 3)] = solver(dt, Itraj[:, (ii * 3):(ii * 3 + 3)])
    if plot:
        # Plot each trajectory
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_pos = Itraj[:, (ii*3)]
        y_pos = Itraj[:, (ii*3)+1]
        z_pos = Itraj[:, (ii*3)+2]
        color = np.random.rand(3)
        ax.plot(x_pos, y_pos, z_pos, '-', color = color, alpha = 1)
        # ax.plot([Ttraj[ii,0]], [Ttraj[ii,1]], [Ttraj[ii,2]], 'x', color = color)
        # ax.plot([target_estimated[0]], [target_estimated[1]], [target_estimated[2]], '.', color=color)
        plt.show()

    with open('data_set_1.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(Itraj)

    print("Data set saved.")

def create_trajectory(dt = 0.01, time_interval=(0, 10), initial_speed_interval=(0, 10),
                      initial_angleAZ_interval=(15, 165), initial_angleAL_interval=(15, 165),
                      noise=1, g=9.81):
    # create_trajectories receives a bunch of trajectory parameters, and returns a trajectory as a matrix.
    # dt - the time difference between two points in the trajectory (observation points)
    # time_interval - the duration of the trajectory (default- flies from time 0 to time 10s)
    # initial_speed_interval - the interval in which the initial speed is randomly sampled
    # initial_angleAZ_interval - the interval in which the initial trajectory azimuth is randomly sampled
    # initial_angleAL_interval - the interval in which the initial trajectory elevation is randomly sampled
    # noise - the standard deviation of random noise added to the trajectory
    # g - the gravitational constant

    t = np.arange(time_interval[0], time_interval[1]+dt, dt)
    n_timesteps = len(t)

    initial_speed = np.random.uniform(initial_speed_interval[0], initial_speed_interval[1])
    initial_angleAZ_interval = np.deg2rad(initial_angleAZ_interval)
    initial_angleAL_interval = np.deg2rad(initial_angleAL_interval)
    initial_direction_AZ = np.random.uniform(initial_angleAZ_interval[0], initial_angleAZ_interval[1])
    initial_direction_AL = np.random.uniform(initial_angleAL_interval[0], initial_angleAL_interval[1])

    x0 = 0
    y0 = 0
    z0 = 0

    v0z = initial_speed * np.sin(initial_direction_AL)
    v0xy = initial_speed * np.cos(initial_direction_AL)
    v0x = v0xy * np.cos(initial_direction_AZ)
    v0y = v0xy * np.sin(initial_direction_AZ)

    trajectory_x = x0 + v0x * t + np.random.normal(size = n_timesteps) * noise
    trajectory_y = y0 + v0y * t + np.random.normal(size = n_timesteps) * noise
    trajectory_z = z0 + v0z * t + 0.5 * (-g) * t**2 + np.random.normal(0, 1, size = n_timesteps) * noise

    t_target = 2*v0z/g
    x_traget = x0 + v0x * t_target
    y_traget = y0 + v0y * t_target
    z_target = 0
    target_POS = np.asarray([x_traget, y_traget, z_target])
    traj_vec = np.vstack((np.vstack((trajectory_x, trajectory_y)), trajectory_z)).T
    return traj_vec, target_POS


def solver(dt, trajectory):
    xtraj = trajectory[:,0]
    ytraj = trajectory[:,1]
    ztraj = trajectory[:,2]

    t = np.arange(0, int((len(xtraj) - 1) * dt) + dt, dt)
    px = np.polyfit(t, xtraj, 1)
    ppx = np.poly1d(px)
    py = np.polyfit(t, ytraj, 1)
    ppy = np.poly1d(py)
    pz = np.polyfit(t, ztraj, 2)
    ppz = np.poly1d(pz)

    dd = np.sqrt( pz[1]**2-4*pz[0]*pz[2])
    t1 = (-pz[1] + dd) / (2*pz[0])
    t2 = (-pz[1] - dd) / (2*pz[0])
    t_target = np.max([t1,t2])

    return np.asarray([ppx(t_target), ppy(t_target), ppz(t_target)])


def RMS(val1, val2):
    return np.sqrt(np.mean((val2 - val1)**2))


def error(val1, val2):
    return np.sqrt(np.sum((val1 - val2)**2))

def my_plot(Itraj, Ttraj, predicted_traj, predicted_hit_pos, n_time_steps):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot([Ttraj[0, 0]], [Ttraj[0, 1]], [Ttraj[0, 2]], 'rx', alpha=1, label='Real target')
    ax.plot(Itraj[:, 0], Itraj[:, 1], Itraj[:, 2], 'b-', alpha=1, label='Observed trajectory')

    ax.plot(predicted_traj[n_time_steps+1:, 0], predicted_traj[n_time_steps+1:, 1],
            predicted_traj[n_time_steps+1:, 2], 'g-', alpha=1, label='Predicted trajectory')
    ax.plot([predicted_hit_pos[0]], [predicted_hit_pos[1]], [predicted_hit_pos[2]], 'or', alpha=1, label='Predicted target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    miss_distance = np.sqrt((Ttraj[0, 0] - predicted_hit_pos[0])**2 + (Ttraj[0, 1] - predicted_hit_pos[1])**2)
    target_distance = np.sqrt(Ttraj[0, 0]**2 + Ttraj[0, 1]**2)
    error = miss_distance / target_distance
    ax.set_title("Estimated target error: {:.2f}%".format(error*100))
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    plt.savefig('plot_10.png')
    plt.show()

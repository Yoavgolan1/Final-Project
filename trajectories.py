import numpy as np

class Trajectories:
    def __init__(self):
        return


def create_trajectory(n_timesteps=200, duration=10, initial_speed_interval=(0, 10),
                      initial_angle_interval=(15, 165), noise=2, g=9.81):

    t = np.linspace(0, duration, n_timesteps)
    initial_angle_interval = np.deg2rad(initial_angle_interval)
    initial_speed = np.random.uniform(initial_speed_interval[0], initial_speed_interval[1])
    initial_direction = np.random.uniform(initial_angle_interval[0], initial_angle_interval[1])

    x0 = 0
    y0 = 0

    v0x = initial_speed * np.cos(initial_direction)
    v0y = initial_speed * np.sin(initial_direction)

    trajectory_x = np.zeros(n_timesteps)
    trajectory_y = np.zeros(n_timesteps)

    for ii in range(n_timesteps):
        trajectory_x[ii] = x0 + v0x * t[ii] + np.random.uniform(-noise, noise)
        trajectory_y[ii] = y0 + v0y * t[ii] + 0.5 * (-g) * t[ii]**2 + np.random.uniform(-noise, noise)
    return (trajectory_x, trajectory_y)


def split_trajectory(traj, split_ratio=0.8):
    traj_x = traj[0]
    traj_y = traj[1]
    n_timesteps = len(traj_x)
    input_length = np.round(split_ratio * n_timesteps).astype(int)
    output_length = n_timesteps - input_length

    input = np.zeros(input_length * 2)
    for ii in range(input_length):
        input[ii] = traj_x[ii]
        input[ii + input_length] = traj_y[ii]

    output = np.zeros(output_length * 2)
    ii = 0
    for jj in range(input_length, n_timesteps):
        output[ii] = traj_x[jj]
        output[ii + output_length] = traj_y[jj]
        ii += 1

    return input, output


t_x = create_trajectory(20)

mm, nn = split_trajectory(t_x)
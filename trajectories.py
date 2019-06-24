import numpy as np
import matplotlib.pyplot as plt

class Trajectories:
    def __init__(self):
        return

def create_trajectory(n_timesteps=200, duration=10, initial_speed_interval=(0, 10),
                      initial_angle_interval=(15, 165), noise=.2, g=9.81):
    # create_trajectories receives a bunch of trajectory parameters, and returns a trajectory as a tuple of
    # x and y coordinates. n_timesteps is the number of points in the trajectory, duration is the total time of
    # flight, i.e. the trajectory starts at time 0 and ends at time 'duration'. initial_speed_interval is the interval
    # within which the initial speed of the trajectory is selected. initial_angle_interval is the interval within
    # which the initial trajectory angle is selected. noise is the magnitude of randomized noise in the trajectory. g
    # is the gravitational constant. The trajectory originates from 0,0.

    t = np.linspace(0, duration, n_timesteps)
    initial_angle_interval = np.deg2rad(initial_angle_interval)
    initial_speed = np.random.uniform(initial_speed_interval[0], initial_speed_interval[1])
    initial_direction = np.random.uniform(initial_angle_interval[0], initial_angle_interval[1])

    x0 = 0
    y0 = 0

    v0x = initial_speed * np.cos(initial_direction)
    v0y = initial_speed * np.sin(initial_direction)

    # trajectory_x = np.zeros(n_timesteps)
    # trajectory_y = np.zeros(n_timesteps)

    trajectory_x = x0 + v0x * t + np.random.normal(size = n_timesteps) * noise
    trajectory_y = y0 + v0y * t + 0.5 * (-g) * t**2 + np.random.normal(size = n_timesteps) * noise
    # for ii in range(n_timesteps):
    #     trajectory_x[ii] = x0 + v0x * t[ii] + np.random.uniform(-noise, noise)
    #     trajectory_y[ii] = y0 + v0y * t[ii] + 0.5 * (-g) * t[ii]**2 + np.random.uniform(-noise, noise)
    return (trajectory_x, trajectory_y)


def split_trajectory(traj, split_ratio=0.8):
    # split_trajectory recieves a trajectory, such as produced by create_trajectory, with a split ratio, and returns
    # two vectors that represent two parts of the trajectory. The split ration decides the length of the respective
    # returned results. i.e., if the trajectory length is 100 timesteps, and the split ratio is 0.8, the first returned
    # value "input" will be a trajectory that is 80 time steps long, and the second returned value "output" will be
    # a trajectory 20 time steps long. The "input" and "output" trajectories are not tuples of x,y coordinates, but
    # rather the x,y coordinates are appended to a single vector.
    # I.e., "input" = [x0, x1, x2, ..., x80, y0, y1, ..., y80], "output" = [x81, x82, ..., x100, y81, y82, ..., y100]

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

def plot(traj_x, traj_y):

    plt.plot(traj_x, traj_y, 'o-r')
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory')
    # plt.show()

def inBasket(alpha = 45, vel = 5):

    Basket_CPos = [10, 5]

#t_x = create_trajectory(20)

#mm, nn = split_trajectory(t_x)
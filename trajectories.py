import numpy as np
import matplotlib.pyplot as plt

class Trajectories:
    def __init__(self):
        return

def create_trajectory(n_timesteps=200, duration=10, initial_speed_interval=(0, 10),
                      initial_angle_interval=(15, 165), noise=.2, g=9.81, basket=[(2, 1), (8, 1)]):
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

    success = inBasket(v0x=v0x, v0y=v0y, basket=basket, g=g)
    return (trajectory_x, trajectory_y), success


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


def inBasket(v0x=2, v0y=10, basket=[(3, 2), (4, 2)], g=9.81):
    xt1 = basket[0][0]
    yt1 = basket[0][1]
    xt2 = basket[1][0]
    yt2 = basket[1][1]

    line_eq_m = (yt1-yt2)/(xt1-xt2)
    line_eq_n = -line_eq_m*xt1 + yt1

    para_eq_a = 0.5 * (-g) / (v0x**2)
    para_eq_b = v0y/v0x
    para_eq_c = 0

    new_para_eq_a = para_eq_a
    new_para_eq_b = para_eq_b - line_eq_m
    new_para_eq_c = para_eq_c - line_eq_n

    if (new_para_eq_b**2 - 4 * new_para_eq_a * new_para_eq_c) < 0:  # There are no coinciding points
        #print("no coincidence")
        return 0
    else:  # There are one or two coinciding points
        x1 = (-new_para_eq_b + np.sqrt(new_para_eq_b**2 - 4 * new_para_eq_a * new_para_eq_c))/(2*new_para_eq_a)
        y1 = para_eq_a * x1**2 + para_eq_b * x1 + para_eq_c

        x2 = (-new_para_eq_b - np.sqrt(new_para_eq_b ** 2 - 4 * new_para_eq_a * new_para_eq_c)) / (2 * new_para_eq_a)
        y2 = para_eq_a * x2 ** 2 + para_eq_b * x2 + para_eq_c

    x_low = np.min([xt1, xt2])
    x_high = np.max([xt1, xt2])
    y_low = np.min([yt1, yt2])
    y_high = np.max([yt1, yt2])

    if ((x1 >= x_low) and (x1 <= x_high)) and ((y1 >= y_low) and (y1 <= y_high)):
        #print("in")
        return 1
    elif ((x2 >= x_low) and (x2 <= x_high)) and ((y2 >= y_low) and (y2 <= y_high)):
        #print("in")
        return 1
    else:
        #print("out")
        return 0


t_x, s = create_trajectory(20)

print(s)
#inBasket(v0x=2, v0y = 10)
#mm, nn = split_trajectory(t_x)
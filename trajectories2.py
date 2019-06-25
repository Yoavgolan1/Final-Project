import numpy as np
import matplotlib.pyplot as plt

class Trajectories:
    def __init__(self):
        return

def create_trajectory(dt = 0.01, time_interval=(0, 10), initial_speed_interval=(0, 10),
                      initial_angleAZ_interval=(15, 165), initial_angleAL_interval=(15, 165),
                      noise=.2, g=9.81):
    # create_trajectories receives a bunch of trajectory parameters, and returns a trajectory as a tuple of
    # x and y coordinates. n_timesteps is the number of points in the trajectory, duration is the total time of
    # flight, i.e. the trajectory starts at time 0 and ends at time 'duration'. initial_speed_interval is the interval
    # within which the initial speed of the trajectory is selected. initial_angle_interval is the interval within
    # which the initial trajectory angle is selected. noise is the magnitude of randomized noise in the trajectory. g
    # is the gravitational constant. The trajectory originates from 0,0.

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
    trajectory_z = z0 + v0z * t + 0.5 * (-g) * t**2 + np.random.normal(size = n_timesteps) * noise

    # success = inBasket(v0x=v0x, v0y=v0y, basket=basket, g=g)
    t_target = 2*v0z/g
    x_traget = x0 + v0x * t_target
    y_traget = y0 + v0y * t_target
    z_target = 0
    target_POS = np.asarray([x_traget, y_traget, z_target])
    # print("v0x={}\tv0y={}\tsuccess={}".format(v0x, v0y,success))
    # return np.append(trajectory_x, trajectory_y, trajectory_z).reshape((1,n_timesteps*3)), target_POS
    traj_vec = np.asarray([trajectory_x, trajectory_y, trajectory_z]).reshape((1, n_timesteps * 3))
    return traj_vec, target_POS

def solver(dt, trajectory):
    n = int(len(trajectory)/3)
    xtraj = trajectory[:n]
    ytraj = trajectory[n:2*n]
    ztraj = trajectory[2*n:]

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

def RMS(t1,t2):
    return np.sqrt(np.mean((t1 - t2)**2))
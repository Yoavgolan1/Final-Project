import numpy as np
import trajectories2 as trj
from lstm_predictor import LSTM_Predictor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create training set
n_trajectories = 2
dt = 0.1
time_interval = (0,90)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 60)
initial_angleAL_interval = (45, 50)
trj.create_dataSet(n_trajectories = n_trajectories, dt = dt, time_interval=time_interval,
                   initial_speed_interval=initial_speed_interval, initial_angleAZ_interval=initial_angleAZ_interval,
                   initial_angleAL_interval=initial_angleAL_interval,
                   noise = 20, g=9.81, plot = False)


inout_size = 10
outpur_size = 2
LSTM = LSTM_Predictor(inout_size, outpur_size)
# LSTM.train('data_set_1.csv', epochs=10)
# my_weights = LSTM.save('weights')

LSTM.load('weights')

## Test
dt = 0.1
time_interval = (0,90)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 60)
initial_angleAL_interval = (45, 50)
n_time_steps = int((time_interval[1] - time_interval[0]) / dt + 1)
Itraj = np.zeros((n_time_steps, 3))
Ttraj = np.zeros((3, 3))
Itraj, Ttraj[0,:] = trj.create_trajectory(dt = dt, time_interval = time_interval,
                                                       initial_speed_interval = initial_speed_interval,
                                                       initial_angleAZ_interval = initial_angleAZ_interval,
                                                       initial_angleAL_interval = initial_angleAL_interval,
                                                       noise = 20)


def test_and_track(inout_size, outpur_size, Itraj):
    print("testing")
    hit = False

    while(hit == False):
        input = Itraj[:-inout_size, :]
        predict = LSTM.test(input)
        i = np.where(predict[:,2]>0)[0]      # search where Z first time is negative

        if i.size > 0:
            hit = True
            print("Hit position in: X={:.2f}\tX={:.2f}\tX={:.2f}".format(predict[i[0], 0], predict[i[0], 1], predict[i[0], 2]))
        else:
            Itraj = np.vstack((Itraj, predict[0, :]))

def print(Itraj, Ttraj, predicted_pos):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(Itraj[:,0], Itraj[:,1], Itraj[:,2], 'b-', alpha=1)
    ax.plot(Ttraj[0, 0], Ttraj[0, 1], Ttraj[0, 2], 'b-', alpha=1)
    plt.show()

test_and_track(inout_size, outpur_size, Itraj)

print("end")
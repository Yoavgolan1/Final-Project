import numpy as np
import trajectories2 as trj
from lstm_predictor import LSTM_Predictor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create training set
n_trajectories = 1
dt = 0.1
time_interval = (0,90)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 60)
initial_angleAL_interval = (45, 50)
trj.create_dataSet(n_trajectories = n_trajectories, dt = dt, time_interval=time_interval,
                   initial_speed_interval=initial_speed_interval, initial_angleAZ_interval=initial_angleAZ_interval,
                   initial_angleAL_interval=initial_angleAL_interval,
                   noise = 0, g=9.81, plot = False)


inout_size = 10
output_size = 10
LSTM = LSTM_Predictor(inout_size, output_size)
# LSTM.train('data_set_1.csv', epochs=100)
# LSTM.save('weights')
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
                                                       noise = 0)

def test_and_track(inout_size, Itraj):
    # print("testing")
    hit = False
    predicted_hit_pos = np.zeros((1,3))
    iter = 0
    while(hit == False):
        iter += 1
        input = Itraj[-inout_size:, :]
        predict = LSTM.test(input)
        i = np.where(predict[:,2]<0.1)[0]      # search where Z first time is negative

        if i.size > 0 or iter == 10:
            hit = True
            # print("Hit position in: X={:.2f}\tY={:.2f}\tZ={:.2f}".format(predict[i[0], 0], predict[i[0], 1], predict[i[0], 2]) )
            # predicted_hit_pos = predict[i[0], :]
        else:
            Itraj = np.vstack((Itraj, predict[0, :]))

        print(Itraj[-1,:])

    return Itraj, predicted_hit_pos


def my_plot(Itraj, Ttraj, predicted_traj, predicted_hit_pos):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot([Ttraj[0, 0]], [Ttraj[0, 1]], [Ttraj[0, 2]], 'rx', alpha=1)
    ax.plot(Itraj[:, 0], Itraj[:, 1], Itraj[:, 2], 'b-', alpha=1)

    ax.plot(predicted_traj[901:, 0], predicted_traj[901:, 1], predicted_traj[901:, 2], 'g.', alpha=1)
    # ax.plot([predicted_hit_pos[0]], [predicted_hit_pos[1]], [predicted_hit_pos[2]], 'or', alpha=1)
    plt.show()

predicted_traj, predicted_hit_pos = test_and_track(inout_size, Itraj)
my_plot(Itraj, Ttraj, predicted_traj, predicted_hit_pos)
print("end")
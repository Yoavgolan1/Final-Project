import numpy as np
import trajectory_functions as trj
from lstm_predictor import LSTM_Predictor


# --------------------------------------------- Create Training Set ---------------------------------------------- #
n_trajectories = 200
dt = 0.5
time_interval = (0, 120)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 45)
initial_angleAL_interval = (40, 45)

n_time_steps = int((time_interval[1] - time_interval[0]) / dt + 1)

trj.create_dataSet(n_trajectories=n_trajectories, dt=dt, time_interval=time_interval,
                   initial_speed_interval=initial_speed_interval, initial_angleAZ_interval=initial_angleAZ_interval,
                   initial_angleAL_interval=initial_angleAL_interval,
                   noise=0.1, g=9.81, plot=False)

# ------------------------------------------------- Train ----------------------------------------------------- #

input_size = 3
output_size = 1

LSTM = LSTM_Predictor(input_size, output_size)
# Uncomment the next two lines to train, leave commented to load saved weights
# LSTM.train('data_set_1.csv', epochs=50)
# LSTM.save('weights')
LSTM.load('weights')

# ------------------------------------------------- Test ----------------------------------------------------- #

single_traj_time_interval = (0, 60)
single_traj_n_time_steps = int((single_traj_time_interval[1] - single_traj_time_interval[0]) / dt + 1)
Itraj = np.zeros((n_time_steps, 3))
Ttraj = np.zeros((3, 3))
Itraj, Ttraj[0, :] = trj.create_trajectory(dt=dt, time_interval=single_traj_time_interval,
                                                       initial_speed_interval=initial_speed_interval,
                                                       initial_angleAZ_interval=initial_angleAZ_interval,
                                                       initial_angleAL_interval=initial_angleAL_interval,
                                                       noise=0)


def test_and_track(inout_size, Itraj, Ttraj):
    # print("testing")
    hit = False
    predicted_hit_pos = np.zeros((1, 3))
    iter = 0
    while(hit == False):
        iter += 1
        x_input = Itraj[-inout_size:, :]
        predict = LSTM.test(x_input)
        i = np.where(predict[:, 2] < 0.1)[0]      # search where Z first time is negative
        if i.size > 0 or iter == 600:
            hit = True
            if i.size > 0:
                predicted_hit_pos = predict[i[0], :]
        else:
            Itraj = np.vstack((Itraj, predict[0, :]))
    error = np.mean(np.abs((Ttraj[0, 1:2]-predicted_hit_pos[0:2])/Ttraj[0, 1:2])*100)
    print("Estimated target error: {:.2f}%".format(error))
    return Itraj, predicted_hit_pos


predicted_traj, predicted_hit_pos = test_and_track(input_size, Itraj, Ttraj)
trj.my_plot(Itraj, Ttraj, predicted_traj, predicted_hit_pos, single_traj_n_time_steps)
print("boom")

import numpy as np
import trajectories2 as trj
import lstm_predictor as LSTM

# Create training set
n_trajectories = 100
dt = 0.1
time_interval = (0,90)
initial_speed_interval = (650, 750)
initial_angleAZ_interval = (30, 60)
initial_angleAL_interval = (45, 50)
trj.create_dataSet(n_trajectories = n_trajectories, dt = dt, time_interval=time_interval,
                   initial_speed_interval=initial_speed_interval, initial_angleAZ_interval=initial_angleAZ_interval,
                   initial_angleAL_interval=initial_angleAL_interval,
                   noise = 20, g=9.81, plot = False)

my_lstm = LSTM.LSTM_Predictor(10,2)
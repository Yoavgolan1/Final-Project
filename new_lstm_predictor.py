# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
import tensorflow as tf
from tqdm import tqdm
import numpy as np


my_data = np.genfromtxt('newfile.csv', delimiter=',')
my_data2 = np.genfromtxt('newfile2.csv', delimiter=',')
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset3 = hstack((in_seq1, in_seq2, out_seq))

#print(dataset3)
X3, y3 = split_sequences(dataset3, 4, 1)

#print(X3.shape)

dataset = my_data
dataset2 = my_data2

# choose a number of time steps
n_steps_in, n_steps_out = 10, 10
# covert into input/output


X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X2, y2 = split_sequences(dataset2, n_steps_in, n_steps_out)
#print(dataset.shape)
#print(X.shape)
#print(y.shape)
#print(y[-1, -1, :])
# the dataset knows the number of features, e.g. 3
n_features = X.shape[2]

# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(tf.keras.layers.RepeatVector(n_steps_out))
model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
#model.fit(X, y, epochs=300, verbose=0)

epochs = 3000
X_batches = [X, X2]
y_batches = [y, y2]
n_batches = 2
for i in tqdm(range(epochs)):
    for j in range(n_batches-1):
        #model.train_on_batch(X_batches[j], y_batches[j])
        xxx = 1
        #print("RMS:", testerror(traj=my_data, model=model, n_steps_in=n_steps_in, n_steps_out=n_steps_out))

# demonstrate prediction
#x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
#x_input = array([my_data[-3, :], my_data[-2, :], my_data[-1, :]])
#x_input = x_input.reshape((1, n_steps_in, n_features))
#print(x_input)
#yhat = model.predict(x_input, verbose=0)
##print(yhat)

#print(dataset)

def testerror(traj, model, n_steps_in, n_steps_out):
    X, y = split_sequences(traj, n_steps_in, n_steps_out)
    x_input = np.array([])
    for i in range(n_steps_in):
        x_input = np.append(x_input, traj[-(n_steps_in + n_steps_out - i), :])
    x_input = x_input.reshape((1, n_steps_in, 3))
    yhat = model.predict(x_input, verbose=0)
    last_real_point = y[-1, -1, :]
    last_predicted_point = yhat[-1, -1, :]
    err = last_real_point - last_predicted_point
    sserr = np.sqrt(np.sum(err**2) / 3)

    #print(last_real_point)
    #print(last_predicted_point)
    error = sserr
    return error


testerror(traj=my_data, model=model, n_steps_in=n_steps_in, n_steps_out=n_steps_out)


def read_trajectories(filename):
    my_data = np.genfromtxt(filename, delimiter=',')
    n_trajectories = int((my_data.shape[1]/3))
    trajectory_length = int(my_data.shape[0])
    trajectories = np.zeros((n_trajectories, trajectory_length, 3))
    for ii in range(n_trajectories):
        trajectories[ii] = my_data[:, ii*3:ii*3+3]
    return trajectories


def train_lstm(model, input_trajectories, epochs, n_steps_in, n_steps_out):
    report = int(epochs / 10)
    n_trajectories = input_trajectories.shape[0]
    first_X, first_y = split_sequences(input_trajectories[0, :, :], n_steps_in, n_steps_out)
    X_batches = []
    y_batches = []
    #X_batches = np.array((n_trajectories, first_X.shape[0], first_X.shape[1], first_X.shape[2]))
    #y_batches = np.array((n_trajectories, first_y.shape[0], first_y.shape[1], first_y.shape[2]))
    #print(first_X.shape)

    for i in range(n_trajectories):
        this_trajectory = input_trajectories[i, :, :]
        this_X, this_y = split_sequences(this_trajectory, n_steps_in, n_steps_out)
        X_batches.append(this_X)
        y_batches.append(this_y)


    for i in tqdm(range(epochs)):
        for j in range(n_trajectories):
            model.train_on_batch(X_batches[j], y_batches[j])
            xxx = 1

        if (i % report) == 0:
            sse = 0
            for j in range(n_trajectories):
                sse += testerror(traj=input_trajectories[j, :, :], model=model, n_steps_in=n_steps_in, n_steps_out=n_steps_out)**2
            sse = np.sqrt(sse/n_trajectories)
            print(i, "/", epochs, " RMS:", sse)


new_trajs = read_trajectories('input_trajectories.csv')

old_trajs = read_trajectories('newfile.csv')


train_lstm(model=model, input_trajectories=new_trajs, epochs=epochs, n_steps_in=n_steps_in, n_steps_out=n_steps_out)



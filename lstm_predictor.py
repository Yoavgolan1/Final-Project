import tensorflow as tf
import numpy as np
from tqdm import tqdm


class LSTM_Predictor:
    def __init__(self, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = 3  # Three coordinates in euclidean space
        self.construct_lstm_model()

    def __str__(self):
        return 'An LSTM object that predicts {self.n_steps_out} trajectory steps' \
               ' forward by examining {self.n_steps_in} trajectory steps.'.format(self=self)

    def split_sequences(self, sequences):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def read_trajectories(self, filename):
        my_data = np.genfromtxt(filename, delimiter=',')
        self.n_trajectories = int((my_data.shape[1] / 3))
        self.trajectory_length = int(my_data.shape[0])
        trajectories = np.zeros((self.n_trajectories, self.trajectory_length, 3))
        for ii in range(self.n_trajectories):
            trajectories[ii] = my_data[:, ii * 3:ii * 3 + 3]
        return trajectories

    def construct_lstm_model(self, n_hidden_1=200, n_hidden_2=200):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(n_hidden_1, activation='relu', input_shape=(self.n_steps_in, self.n_features)))
        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        model.add(tf.keras.layers.LSTM(n_hidden_2, activation='relu', return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features)))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train(self, training_set_filename, epochs=300, report=999):
        if report == 999:
            report = int(epochs / 10)

        input_trajectories = self.read_trajectories(training_set_filename)

        n_trajectories = self.n_trajectories
        X_batches = []
        y_batches = []

        for i in range(n_trajectories):
            this_trajectory = input_trajectories[i, :, :]
            this_X, this_y = self.split_sequences(this_trajectory)
            X_batches.append(this_X)
            y_batches.append(this_y)

        for i in tqdm(range(epochs)):
            for j in range(n_trajectories):
                self.model.train_on_batch(X_batches[j], y_batches[j])

            if (i % report) == 0:
                sse = 0
                for j in range(n_trajectories):
                    sse += self.testerror(traj=input_trajectories[j, :, :]) ** 2
                sse = np.sqrt(sse / n_trajectories)
                print(i, "/", epochs, " RMS:", sse)

    def testerror(self, traj):
        X, y = self.split_sequences(traj)
        x_input = np.array([])
        for i in range(self.n_steps_in):
            x_input = np.append(x_input, traj[-(self.n_steps_in + self.n_steps_out - i), :])
        x_input = x_input.reshape((1, self.n_steps_in, self.n_features))
        yhat = self.model.predict(x_input, verbose=0)
        last_real_point = y[-1, -1, :]
        last_predicted_point = yhat[-1, -1, :]
        err = last_real_point - last_predicted_point
        sserr = np.sqrt(np.sum(err ** 2) / self.n_features)

        return sserr

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename, by_name=False)

    def test(self, x_input):
        x_input = x_input.reshape((1, self.n_steps_in, self.n_features))
        y_output = self.model.predict(x_input, verbose=0)

        return y_output[0, :, :]






my_lstm = LSTM_Predictor(3, 20)
my_lstm.train('input_trajectories.csv', epochs=100)

#my_lstm.load('test_weights')
my_weights = my_lstm.save('test_weights')

input_block = np.array([[74.446, 529.7, -100.2279], [75.59, 537.863, -105.072], [76.737, 546.0126, -110.01804]])

next_guess = my_lstm.test(input_block)
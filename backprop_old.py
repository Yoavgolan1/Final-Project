import numpy as np
import math
from tqdm import tqdm
import pickle

niter = 10000  # Number of iterations

# ---------------------------------- Part 0: Creating a Back propagation object  --------------------------------------


class Backprop:
    def __init__(self, n, m, h):
        self.n = n
        self.m = m
        self.h = h
        self.interval = 0.05
        self.weights_IH = self.interval * 2 * np.random.random((n+1, h)) - self.interval
        self.weights_HO = self.interval * 2 * np.random.random((h+1, m)) - self.interval

    def __str__(self):
        return 'A backprop object with {self.n} inputs, {self.m} outputs, and {self.h} hidden units.'.format(self=self)

    def test(self, I):
        vfunc = np.vectorize(self.squash)
        i = np.append(I, 1)
        Hnet = np.array(np.dot(i, self.weights_IH))
        H = np.array(vfunc(Hnet))
        Onet = np.dot(np.append(H, 1), self.weights_HO)
        O = vfunc(Onet)
        return O

    def train(self, input_pattern, target_pattern, iterations, eta, mu, lambada):
        input_rows = input_pattern.shape[0]
        to_append = np.array(np.ones((input_rows, 1)))
        input_pattern = np.hstack((input_pattern, to_append))
        vfunc = np.vectorize(self.squash)
        vfunc2 = np.vectorize(self.squash_prime)
        delta_w_IH_prev = np.array(np.zeros((self.n + 1, self.h)))
        delta_w_HO_prev = np.array(np.zeros((self.h + 1, self.m)))
        print("Training Backprop...")
        for ii in tqdm(range(iterations)):
            delta_w_IH = np.array(np.zeros((self.n + 1, self.h)))
            delta_w_HO = np.array(np.zeros((self.h + 1, self.m)))
            for j in range(input_rows):
                Ij = input_pattern[j, :]
                Hnet = np.array(np.dot(Ij, self.weights_IH))
                H = np.array(vfunc(Hnet))
                Onet = np.dot(np.append(H, 1), self.weights_HO)
                O = vfunc(Onet)
                Tj = target_pattern[j, :]
                del_O = np.array([a*b for a, b in zip(Tj-O, vfunc2(Onet))])
                temp = (np.dot(del_O, self.weights_HO.T))[:-1]
                del_H = [a*b for a, b in zip(temp, vfunc2(Hnet))]
                delta_w_IH += np.outer(Ij.T, del_H)
                delta_w_HO += np.outer(np.append(H, 1).T, del_O)

            weight_change_IH = eta*delta_w_IH + mu*delta_w_IH_prev - lambada * self.weights_IH
            delta_w_IH_prev = delta_w_IH
            delta_w_IH = weight_change_IH
            self.weights_IH += delta_w_IH/input_rows

            weight_change_HO = eta * delta_w_HO + mu*delta_w_HO_prev - lambada * self.weights_HO
            delta_w_HO_prev = delta_w_HO
            delta_w_HO = weight_change_HO
            self.weights_HO += delta_w_HO / input_rows

    @staticmethod
    def squash(element):
        return 1 / (1 + math.exp(-element))

    def squash_prime(self, element):
        return self.squash(element) * (1 - self.squash(element))

    def save(self, filename):
        pickle_out = open(filename, "wb")
        pickle.dump({"WIH": self.weights_IH, "WHO": self.weights_HO}, pickle_out)
        pickle_out.close()

    def load(self, filename):
        pickle_in = open(filename, "rb")
        a = pickle.load(pickle_in)
        self.weights_IH = a["WIH"]
        self.weights_HO = a["WHO"]

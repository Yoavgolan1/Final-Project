# Shmulik Edelman, shmulike@post.bgu.ac.il

# n     - Number of inputs
# m     - Number of outputs
# niter - Number of iterations, default = 1000
# h     - Hidden layers, default = 1
# eta   - Learning rate, default = 0.5

import numpy as np
import pickle
import os

class Backprop():
    def __init__(self, n, m, h = 1):
        # self.vec = np.array([1,5])
        self.n = n
        self.m = m
        self.h = h
        self.w_ih = (np.random.random((n + 1, h))) / n
        self.w_ho = (np.random.random((h + 1, m))) / n

    def __str__(self):
        return "Back-prop with {} inputs, {} outputs and {} hidden units".format(self.n, self.m, self.h)

    def sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

    def dsigmoid(self, val):
        return val * (1 - val)

    def test(self, I):
        H = np.dot(np.hstack((I, np.ones((len(I), 1)))), self.w_ih)
        H = self.sigmoid(H)
        O = np.dot(np.hstack((H, np.ones((len(H), 1)))), self.w_ho)
        O = self.sigmoid(O)
        return(O)

    def save(self,bp, ipart, h, eta):
        fileName = './part{}/Part{}_h{}_eta{}.wgt'.format(ipart, ipart, h, np.round(eta,1))
        pickle.dump(bp, open(fileName, "wb"))

    def load(self, ipart, h, eta):
        fileName = './part{}/Part{}_h{}_eta{}.wgt'.format(ipart, ipart, h, np.round(eta,1))
        return pickle.load(open(fileName, "rb"))

    def RMS(self, T, O):
        return np.sqrt(np.mean((T - O)**2))

    def XOR_RMS(self, T, O):
        return np.sqrt(np.mean((T[1::3] - O[1::3]) ** 2))

    def train(self, I, T, niter=1000, eta = 0.5, mu  =0, rms_flag = 0, hidden = 0, report = 1):

        rms = np.zeros(niter)

        for n in range(niter):
            O_full = np.zeros((I.shape[0], self.m))
            for k in range(I.shape[1]):
                i = np.append(I[k,:], 1)
                H_net = np.dot(i, self.w_ih)
                H = self.sigmoid(H_net)
                H_ = np.append(H, 1)
                O_net = np.dot(H_, self.w_ho)
                O = self.sigmoid(O_net)
                O_full[k,:] = O

                delta_o = (T[:] - O) * self.dsigmoid(O)
                delta_h = (np.dot(delta_o, self.w_ho.T))[:-1]
                delta_h = delta_h * self.dsigmoid(H)

                dw_ih = np.outer(i, delta_h)
                dw_ho = np.outer(H_, delta_o)

                self.w_ih += eta * dw_ih
                self.w_ho += eta * dw_ho

            rms[n] = self.RMS(T, O_full)
            rms = rms.astype('float')
            if report and n%10==0 and n>0:
                print("{}/{}: {:.6f}".format(n, niter, rms[n]))

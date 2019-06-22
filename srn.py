import numpy as np
from sys import stdout

class SRN:

    def __init__(self, n, h, m):

        self.wih = np.random.randn(n+h+1, h)
        self.who = np.random.randn(h+1, m)

    def train(self, inputs, targets, niter=1000, eta=0.05, report=100):

        for i in range(niter):

            # Sum squared error
            sse = 0

            # Initialize state-layer activations to zero
            ah = np.zeros(self.wih.shape[1])

            for ai, t in zip(inputs, targets):

                ainew = np.append(np.append(ah, ai), 1)

                hnet = np.dot(ainew, self.wih)
                ah   = SRN._f(hnet)

                onet = np.dot(np.append(ah, 1), self.who)
                ao   = SRN._f(onet)

                eo = t - ao
                do = eo * SRN._df(ao)

                # This is back-prop!
                eh = np.dot(do, self.who.T)[:-1]
                dh = eh * SRN._df(ah)

                # Delta rule: adjust weights in proportion to error
                self.wih += eta * np.outer(ainew, dh)
                self.who += eta * np.outer(np.append(ah,1), do)

                # Accumulate sum squared error
                sse += np.sum(eo**2)

            # Report periodically
            if i%report == 0:

                # Compute RMS error as square root of sum squared error, after dividing by
                # number of patterns times number of outputs
                print(i, "/", niter, " RMS:", np.sqrt(sse/(len(inputs)*self.who.shape[1])))

                # Force immediate print
                stdout.flush()

    def test(self, inputs):

        ah = np.zeros(self.wih.shape[1])
        output = np.zeros((len(inputs), self.who.shape[1]))
        counter = 0
        for ai in inputs:

            ainew = np.append(np.append(ah, ai), 1)

            hnet = np.dot(ainew, self.wih)
            ah   = SRN._f(hnet)

            onet = np.dot(np.append(ah, 1), self.who)
            ao   = SRN._f(onet)
            output[counter] = ao
            counter += 1
        return output

    def _f(x):

        return 1 / (1 + np.exp(-x))

    def _df(x):

        return x * (1 - x)


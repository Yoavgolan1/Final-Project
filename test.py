import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0,1,1000)*15

print(np.std(x))
plt.figure()
plt.hist(x)
plt.show()
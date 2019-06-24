import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

Ttraj = True
print() if Ttraj == True else marker = '--g'

# line_style = '--'
# line_color = 'm'
# plt.figure()
# plt.plot(x, y, linestyle = line_style, color = line_color, alpha=0.5)
#
# basket = np.asarray([[10,0], [15,0]])
# plt.plot(basket[:,0], basket[1,:], 'rd-')
#
# plt.show()

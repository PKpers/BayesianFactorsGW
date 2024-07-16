import matplotlib.pyplot as plt
import numpy as np

x_t = np.linspace(0.5, 7)

x = [0.5, 2, 5, 7]
y = [18.32, 6.35, 5.02, 4.49]

plt.plot(x, y, 'b--')
plt.scatter(x, y)
plt.ylabel('linear/quadratic')
plt.xlabel(r'$\sigma$')
plt.show()

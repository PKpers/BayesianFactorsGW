import random as rand
import math as m
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as Int
pi = m.pi

def f(x):
    #the PDF 
    sigma = 0.5
    mu = 0
    c = 1/np.sqrt(2*pi*sigma**2)
    return c*np.exp(-0.5*((x-mu)/sigma)**2 )


x_eval = np.linspace(-pi/2, pi/2, 100) 
f_eval = f(x_eval)


plt.figure()
plt.plot(x_eval, f_eval, label='posterior')

f_max = f(0)
#g_max = g(0)
sampled = list()

x0 = -1 # the initial position
x_last = x0
while len(sampled) != 10000:
    xi = rand.uniform(-pi/2, pi/2) # The proposal distribution is uniform
    p = f(xi)/f(x_last) # in the uniform case Q(xi) = Q(x_{i-1})
    T = min(1, p)
    u = np.random.random_sample()
    if u <= T:
        sampled.append(xi)
        x_last = xi
    #
#
plt.hist(sampled, density=True, label = 'metropolis-hastings samples')
plt.legend(loc='best')
plt.show()



exit()

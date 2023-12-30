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

def g(x):
    return np.cos(x)


x_eval = np.linspace(-pi/2, pi/2, 100) 
g_eval = g(x_eval)
f_eval = f(x_eval)

plt.figure()
plt.plot(x_eval, f_eval, label='posterior')
plt.plot(x_eval, g_eval, label=r'proposal distribution: $\mathrm{g(x)=cos(x)}$')
plt.legend()

f_max = f(0)
g_max = g(0)
sampled = []
sampled_x = []
k = 0

# Draw n iid samples from the proposal g(x) = cos(x)
while len(sampled) != 100000:
    x = rand.uniform(-pi/2, pi/2)
    y = rand.uniform(0, g_max)
    g_ = g(x)
    if y<g_:
        sampled.append(y)
        sampled_x.append(x)
    #
    k += 1
#
sampled_x = np.array(sampled_x)
weights_ = f(sampled_x)/g(sampled_x) # calculate the weights f(x)/g(x)
# According to importance sampling this should approximate the integral of f
print(np.mean(weights_)) 

integral = Int.quad(f, -pi/2, pi/2)
print(integral) #The actual value of the integal \int{f(x)}

counts, bins, weights = plt.hist(sampled_x, 
                                 weights = weights_, 
                                 density=True, 
                                 label = 'histogram of sampled x'
)

plt.scatter(sampled_x, weights_*g(sampled_x), 
            color='red', marker='.', 
            label = 'posterior f, evaluated on the samples'
)
plt.legend(loc='best')
#binwidth = np.diff(bins)
#integral = sum(binwidth*counts)
#print(integral)

plt.show()

exit()

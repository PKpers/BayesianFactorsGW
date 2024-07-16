from numpy import random as rand
from matplotlib import pyplot as plt
from scipy import integrate as Int
import numpy as np
pi = np.pi

def f(x):
    return (1/2)*np.sin(x)

def g(x, mu=pi/2, sigma=1):
    norm = 1/(np.sqrt(2*pi*sigma**2))
    expo = np.exp( -0.5*( (x - mu)/sigma)**2 )
    return norm*expo



x_eval = np.linspace(0,np.pi, 100)
Delta_x = np.abs( x_eval[0] - x_eval[-1])

g_eval = g(x_eval)
f_eval = f(x_eval)


plt.figure()
plt.plot(x_eval, f_eval, color='black', label='posterior')
plt.plot(x_eval, g_eval, color='blue', label ='helper')

sampled = list()

x0 = 0.5 # the initial position
x_acc = x0 
k = 0
while len(sampled) != 1000000:
    k+=1
    x_new = rand.uniform(0,np.pi) # The proposal distribution is uniform
    p = f(x_new)/f(x_acc) # in the uniform case Q(xi) = Q(x_{i-1})
    T = min(1, p)
    u = rand.random_sample()
    if u <= T:
        x_acc = x_new
    sampled.append(x_acc)
    
    #
#
sampled = np.array(sampled)
f_samp = f(sampled)
g_samp = g(sampled)

weights = g_samp/f_samp  
weights_norm = weights/np.sum(weights)
sampled_indices = rand.choice(len(sampled), size=len(sampled), p=weights_norm)
sampled_g = sampled[sampled_indices]


MC_integral = np.mean(weights)
real_integral = Int.quad(g, 0, pi)
best = np.median(sampled_g)

print(MC_integral)
print(real_integral)

#plt.scatter(sampled, f_samp, marker='+', color='black', label='sampled values')
plt.hist(sampled, bins=50, density=True, alpha=0.5, label = 'MC samples for f(x)')
plt.hist(sampled_g, bins=50, density=True, alpha=0.5, label = 'MC samples for g(x)')
plt.axvline(best, color='purple')
plt.legend(loc='best')
plt.show()



exit()

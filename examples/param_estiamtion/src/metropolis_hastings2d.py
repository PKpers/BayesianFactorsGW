from numpy import random as rand
from matplotlib import pyplot as plt
from scipy import integrate as Int
import numpy as np
import seaborn as sns
pi = np.pi


def init_plotting():
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 14
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    
    return



def f(x,y):
    return (1/4)*np.sin(x)*np.sin(y)

def g(x,y, mu=pi/2, sigma=1):
    norm = 1/(np.sqrt(2*pi*sigma**2))
    expo = np.exp( -0.5*( ((x - mu)/sigma )**2 + ((y - mu)/sigma )**2) )
    return norm*expo


init_plotting()
x_eval = np.linspace(0,np.pi, 100)
y_eval = np.linspace(0,np.pi, 100)

#g_eval = g(x_eval, y_eval)
#f_eval = f(x_eval, y_eval)


#plt.figure()
#plt.plot(x_eval, f_eval, color='black', label='posterior')
#plt.plot(x_eval, g_eval, color='blue', label ='helper')

sampled_x, sampled_y = list(), list()

x0, y0 = 0.5, 0.5 # the initial position
x_acc, y_acc = x0, y0 
k = 0
N = 1000
n_samples = 1
while len(sampled_x) != N:
    k+=1
    x_new = rand.uniform(0,np.pi) # The proposal distribution is uniform
    y_new = rand.uniform(0,np.pi) # The proposal distribution is uniform
    p = f(x_new,y_new)/f(x_acc, y_acc) # in the uniform case Q(xi) = Q(x_{i-1})
    T = min(1, p)
    u = rand.random_sample()
    if u <= T:
        x_acc, y_acc = x_new, y_new
        n_samples += 1
    sampled_x.append(x_acc)
    sampled_y.append(y_acc)
    print('Acceptance: {perc} ({samples_N}/{tot_N})'.format(perc=(n_samples/(k+1)), samples_N=n_samples, tot_N=k+1))
    #
#
sampled_x, sampled_y = np.array(sampled_x), np.array(sampled_y)
sampled = np.vstack( (sampled_x, sampled_y) ).T

f_samp = f(sampled_x, sampled_y)
g_samp = g(sampled_x, sampled_y)

weights = g_samp/f_samp  
weights_norm = weights/np.sum(weights)
sampled_indices = rand.choice(N, size=N, p=weights_norm)

sampled_g  = sampled[sampled_indices]
sampled_gx, sampled_gy = sampled_g.T


MC_integral = np.mean(weights)
real_integral = Int.dblquad(g, 0, pi, 0, pi)
best = np.median(sampled_g)

print(MC_integral)
print(real_integral)


#plt.scatter(sampled, f_samp, marker='+', color='black', label='sampled values')
plt.figure()
plt.hist(sampled_gx, bins=50, density=True, alpha=0.5, color='blue', label = 'g(x)')
plt.legend()

plt.figure()
plt.hist(sampled_gy, bins=50, density=True, alpha=0.5, color='black', label = 'g(y)')
plt.legend()

g = sns.JointGrid(x = sampled_gx, y=sampled_gy)
g.plot_joint(sns.histplot, palette='viridis')


g = sns.JointGrid(x = sampled_gx, y=sampled_gy)
g.plot_joint(sns.histplot, palette='viridis')

plt.show()


exit()

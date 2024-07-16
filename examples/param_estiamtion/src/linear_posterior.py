import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, tplquad, quad
pi = np.pi
#############
# Functions #
#############
def init_plotting():
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 10
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


def sample(n_points, x_range, dist):
    '''
    samples from the values of a given distribution
    over a given range. Using rejection sampling

    Arguments:
    - n_points:   integer,
                  number of points to be sampled

    - x_range:    array_like(1d),
                  the domain of the distribution

    - dist:       array_like(1d)
                  the values of the distribution

    Returns:
    - sampled_x:  list(1d),
                  the sampled points
    '''
    sampled = list()
    while len(sampled) != n_points:
        x = rand.uniform(x_range[0], x_range[-1])
        y = rand.uniform(0, max(dist))
        fx = np.interp(x, x_range, dist) #estimate f(x) via interpolation
        if y<fx:
            sampled.append(x)
        #
    #
    return sampled
    

def likelihood_gaussian(a, b, times, datas, sigma_noise):

    num      = (datas - model_lin(times, a, b))**2
    gaussian = 1./np.sqrt(2.*np.pi*sigma_noise**2) * np.exp((-1./2.) * np.sum(num)/(sigma_noise)**2 )

    return gaussian

def help_gauss_2d(ax, bx, a, b, sigma=1 ):
    norm = 1/(np.sqrt(2*pi*sigma**2))
    expo = np.exp( -0.5*( ((ax - a)/sigma )**2 + ((bx - b)/sigma )**2) )
    return norm*expo


def likelihood_gaussian_quad(a, b, c, times, datas, sigma_noise):

    num      = (datas - model_quad(times, a, b, c))**2
    gaussian = 1./np.sqrt(2.*np.pi*sigma_noise**2) * np.exp((-1./2.) * np.sum(num)/(sigma_noise)**2 )

    return gaussian

def help_gauss_3d(ax, bx, cx, a, b, c, sigma=1.5 ):
    norm = 1/(np.sqrt(2*pi*sigma**2))
    expo = np.exp( -0.5*( ((ax - a)/sigma )**2 + ((bx - b)/sigma )**2 + ( (cx - c)/sigma )**2) )
    return norm*expo


def model_lin(x,a,b):

    return a + x * b

def model_quad(x,a,b,c):

    return a + b*x +c*x*x


init_plotting()
###############
# User inputs #
###############

show_data    = 1
zero_noise   = 0

N_data       = 700 # 100, 300, 500, 700, 1000
a_inj, b_inj = 0.5, 1.1
t_min, t_max = 0.0, 1.0
sigma        = 0.2 #0.05, 0.1, 0.2, 0.6, 1

fig_path = '/home/kpapad/puk/examples/param_estiamtion/figures/MC_s02_N{}_'.format(str(N_data))
###############
# Create data #
###############

fig_dat, ax_dat = plt.subplots()
#figsize=(15, 10)

np.random.seed(1)
time      = np.linspace(t_min,t_max,N_data)
model_inj = model_lin(time, a_inj, b_inj)
data      = model_inj

ax_dat.scatter(time, data,   c='black', marker = "+", alpha=0.9, label='Noiseless Data')

if not(zero_noise): data += np.random.normal(0, sigma, size=N_data)

fake_data = open('Fake_data.txt', 'w')
fake_data.write('#Time \t\tData\n')
for i in range(0, N_data):
    fake_data.write('%f\t%f\n' %(time[i], data[i]))
fake_data.close()

if(show_data):
    ax_dat.scatter(time, data,   c='firebrick', alpha=0.9, label='Data')
    #ax_dat.plot(time, model_inj, c='black', linestyle='-', label='Injection')
    ax_dat.legend(loc='best')
    ax_dat.set_title('Gaussian noisy linear data')
    ax_dat.set_xlabel('$Time [s]  $')
    ax_dat.set_ylabel('$Data [AU] $')

##############################
# Calculate linear posterior #
##############################

a_min    = -1
a_max    = 1
b_min    = -1
b_max    = 3
n_points = 500000

a_vec = np.linspace(a_min, a_max, n_points)
b_vec = np.linspace(b_min, b_max, n_points)
#X,Y   = np.meshgrid(a_vec, b_vec) # grid of points

## Define the posterior as a function of the parameters only
prior_l     = (1./(a_max-a_min)) * (1./(b_max-b_min)) # we assume uniforms posteriors
posterior_l = lambda a, b: prior_l*likelihood_gaussian(a, b, time, data, sigma)
proposal_l =  lambda a, b: help_gauss_2d(a, b, a_inj, b_inj)

## Metropolis algorithm. The proposal distribution is uniform in a and b 
k = 0
n_samples = 1
sampled_a, sampled_b = list(), list()
a0, b0 = a_inj, b_inj 
a_acc, b_acc = a0, b0 # acc or accepted

print('Starting the 2d MC sampling...')

while len(sampled_a) != n_points:
    k += 1
    a_new = rand.uniform(a_min, a_max)
    b_new = rand.uniform(b_min, b_max)
    p = proposal_l(a_new,b_new)/proposal_l(a_acc, b_acc) 
    T = min(1, p)
    u = np.random.random_sample()
    if u<=T:
        a_acc, b_acc = a_new, b_new
        n_samples += 1
    sampled_a.append(a_acc)
    sampled_b.append(b_acc)
    print('Acceptance: {perc} ({samples_N}/{tot_N})'.format(perc=(n_samples/(k+1)), samples_N=n_samples, tot_N=k+1))
    #
#
print('... done')

print('Getting samples from the posterior...')

sampled_a, sampled_b = np.array(sampled_a), np.array(sampled_b)
post_samples_l = np.array( [ posterior_l(sampled_a[i], sampled_b[i]) for i in range(len(sampled_a)) ] )
prop_samples_l = proposal_l(sampled_a, sampled_b)

weights = post_samples_l/prop_samples_l  
weights_norm = weights/np.sum(weights)
sampled_indices = rand.choice(n_points, size=n_points, p=weights_norm)

sampled = np.vstack( (sampled_a, sampled_b) ).T

sampled_post_l  = sampled[sampled_indices]
sampled_a, sampled_b = sampled_post_l.T #These are the marginalized posteriors
print('...done')

evidence_l = np.mean(weights)

plt.figure()
counts, a_bins, b_bins, _ = plt.hist2d(sampled_a, sampled_b)

# Find the best estimate for the parameters
a_est = np.median(sampled_a)
a_lower= a_est - np.percentile(sampled_a,5)
a_upper= np.percentile(sampled_a,95) - a_est

b_est = np.median(sampled_b)
b_lower = b_est - np.percentile(sampled_b,5)
b_upper= np.percentile(sampled_b,95) - b_est


# Having sampled a and b from their posteriors, one can consider the model as a
# random variable depending on a and b(also random variables).
# Lets plot the distribution m(a, b) = a + b*x
print('Starting the error bands evaluation...')
linear_dist = [model_lin(time, sampled_a[i], sampled_b[i]) for i in range(len(sampled_a)) ]

upper_bounds = list()
lower_bounds = list()
best_fit     = list()

for t in range(len(time)):
    points = list()
    for line in linear_dist:
        points.append(line[t])
    #
    best = np.median(points)
    lower = np.percentile(points, 5)
    upper = np.percentile(points, 95) 
    
    best_fit.append(best)
    upper_bounds.append(upper)
    lower_bounds.append(lower)
    #
#
print('...done')

print('Making the plots...')
ax_dat.plot(time, best_fit, 'b--',
            label='linear model from best point'
            
)
ax_dat.fill_between(time, lower_bounds, upper_bounds,
                   color = 'blue', alpha = 0.2,
                   label = 'linear: 90% CI')


# plot the posterior
#plt.scatter(X,Y, c=posterior_l, s=100, marker='*', cmap='viridis')
plt.plot(a_inj, b_inj, marker='+', markersize=12, c='black', label='real values')
plt.plot(a_est, b_est, marker='+', markersize=12, c='firebrick',label='estimated values')
plt.xlabel('a')
plt.ylabel('b')
plt.title('$\mathrm{p(a,b | D, H, I)}$')
plt.legend(loc='best')
plt.savefig(fig_path+'linear_posterior1.pdf')

fig_l = plt.figure(figsize=(15, 10))
ax_l = fig_l.subplots(1, 2)

ax_l[0].hist(sampled_a)
ax_l[0].axvline(a_est, color = "red", label='best estimate: {}'.format(round(a_est,2)))
ax_l[0].axvline(a_est - a_lower , color = "blue", label = '5%')
ax_l[0].axvline(a_est + a_upper , color = "blue", label = '95%')
ax_l[0].axvline(a_inj, color='orange', linestyle='--', label='real value')
ax_l[0].set_xlabel('a')
ax_l[0].set_ylabel(r'$P(a|H1,D,I) = \int db P(a,b|H1,D,I)$')
ax_l[0].legend()

ax_l[1].hist(sampled_b)
ax_l[1].axvline(b_est, color = "red", label='best estimate {}'.format(round(b_est,2)))
ax_l[1].axvline(b_est - b_lower , color = "blue", label = '5%')
ax_l[1].axvline(b_est + b_upper , color = "blue", label = '95%')

ax_l[1].axvline(b_inj, color='orange', linestyle='--', label='real value')
ax_l[1].set_xlabel('b')
ax_l[1].set_ylabel(r'$P(b|H2,D,I) = \int da  P(a,b|H1,D,I)$')
ax_l[1].legend()

fig_l.savefig(fig_path+'linear_posterior_marginals1.pdf')

ax_dat.legend(loc='best')

print('...done')
print('inference for the linear model complete')

#################################
# Calculate quadratic posterior #
#################################
print('Starting the quadratic model inference')

c_min    = -2
c_max    = 2
n_points = 1000000

c_vec = np.linspace(c_min, c_max, n_points)

prior_q     = (1./(a_max-a_min)) * (1./(b_max-b_min)) * (1./(c_max-c_min))
posterior_q = lambda a, b, c: prior_q*likelihood_gaussian_quad(a, b, c, time, data, sigma)
proposal_q =  lambda a, b, c: help_gauss_3d(a, b, c, a_inj, b_inj, 0)

## Metropolis algorithm. The proposal distribution is uniform in a and b 
print('Starting the sampling...')

sampled_a, sampled_b, sampled_c = list(), list(), list()
a0, b0, c0 = a_inj, b_inj, 0 
a_acc, b_acc, c_acc = a0, b0, c0
k, n_samples = 0, 1
while len(sampled_a) != n_points:
    k+=1
    a_new = rand.uniform(a_min, a_max)
    b_new = rand.uniform(b_min, b_max)
    c_new = rand.uniform(c_min, c_max)
    
    p  = proposal_q(a_new, b_new, c_new)/proposal_q(a_acc, b_acc, c_acc) 
    T  = min(1, p)
    u  = np.random.random_sample()
    if u<=T:
        a_acc, b_acc, c_acc= a_new, b_new, c_new
        n_samples+=1
        print('Acceptance: {perc} ({samples_N}/{tot_N})'.format(perc=(n_samples/(k+1)), samples_N=n_samples, tot_N=k+1))
        
    sampled_a.append(a_new)
    sampled_b.append(b_new)
    sampled_c.append(c_new)
    #
#
print('...done')

print('Getting samples from the posterior...')
sampled_a, sampled_b, sampled_c = np.array(sampled_a), np.array(sampled_b), np.array(sampled_c)
post_samples_q = np.array( [ posterior_q(sampled_a[i], sampled_b[i], sampled_c[i]) for i in range(n_points) ] )
prop_samples_q = proposal_q(sampled_a, sampled_b, sampled_c)

weights = post_samples_q/prop_samples_q  
weights_norm = weights/np.sum(weights)
sampled_indices = rand.choice(n_points, size=n_points, p=weights_norm)

sampled = np.vstack( (sampled_a, sampled_b, sampled_c) ).T

sampled_post_q  = sampled[sampled_indices]
sampled_a, sampled_b, sampled_c = sampled_post_q.T #These are the marginalized posteriors

print('...done')
evidence_q = np.mean(weights)

post_odds = evidence_l/evidence_q
#ax_dat.text(0.6, 0.55, r'linear/quadratic = {}'.format(round(post_odds,2)))



# Find the best estimate for the parameters
a_est = np.median(sampled_a)
a_lower= a_est - np.percentile(sampled_a,5)
a_upper= np.percentile(sampled_a,95) - a_est

b_est = np.median(sampled_b)
b_lower = b_est - np.percentile(sampled_b,5)
b_upper= np.percentile(sampled_b,95) - b_est

c_est = np.median(sampled_c)
c_lower = c_est - np.percentile(sampled_c,5)
c_upper= np.percentile(sampled_c,95) - c_est

#counts = counts / evidence 

# Having sampled a and b from their posteriors, one can consider the model as a
# random variable depending on a and b(also random variables).
# Lets plot the distribution m(a, b) = a + b*x

print('Starting the error bands evaluation...')

quad_dist = [ model_quad(time, sampled_a[i], sampled_b[i], sampled_c[i]) for i in range(len(sampled_a)) ]

upper_bounds = list()
lower_bounds = list()
best_fit     = list()

for t in range(len(time)):
    points = list()
    for line in quad_dist:
        points.append(line[t])
    #
    best = np.median(points)
    lower = np.percentile(points, 5)
    upper = np.percentile(points, 95) 
    
    best_fit.append(best)
    upper_bounds.append(upper)
    lower_bounds.append(lower)
    #
#
print('...done')
            
print('Making the plots...')

ax_dat.plot(time, best_fit, 'g--',
            label='quadratic model from best point'
            
)

ax_dat.fill_between(time, lower_bounds, upper_bounds,
                   color = 'green', alpha = 0.2,
                   label = 'quadratic: 90% CI')


# plot the marginalized posteriors for the quadtratic case
fig1 = plt.figure(figsize=(18, 10))
ax1 = fig1.subplots(1, 3)

ax1[0].hist(sampled_a)
ax1[0].axvline(a_est, color = "red", label='best estimate: {}'.format(round(a_est,2)))
ax1[0].axvline(a_est - a_lower , color = "blue", label = '5%')
ax1[0].axvline(a_est + a_upper , color = "blue", label = '95%')
ax1[0].axvline(a_inj, color='orange', linestyle='--', label='real value')
ax1[0].set_xlabel('a')
ax1[0].set_ylabel(r'$P(a|H2,D,I) = \int\int db dc P(a,b,c|H2,D,I)$')
ax1[0].legend()

ax1[1].hist(sampled_b)
ax1[1].axvline(b_est, color = "red", label='best estimate: {}'.format(round(b_est,2)))
ax1[1].axvline(b_est - b_lower , color = "blue", label = '5%')
ax1[1].axvline(b_est + b_upper , color = "blue", label = '95%')

ax1[1].axvline(b_inj, color='orange', linestyle='--', label='real value')
ax1[1].set_xlabel('b')
ax1[1].set_ylabel(r'$P(b|H2,D,I) = \int\int da dc P(a,b,c|H2,D,I)$')
ax1[1].legend()

ax1[2].hist(sampled_c)
ax1[2].axvline(c_est, color = "red", label='best estimate: {}'.format(round(c_est,2)))
ax1[2].axvline(c_est - c_lower , color = "blue", label = '5%')
ax1[2].axvline(c_est + c_upper , color = "blue", label = '95%')

ax1[2].axvline(0, color='orange', linestyle='--', label='real value')
ax1[2].set_xlabel('c')
ax1[2].set_ylabel(r'$P(c|H2,D,I) = \int\int da db P(a,b|H2,D,I)$')
ax1[2].legend()
fig1.savefig(fig_path+'quad_posteriors_marginalized1.pdf')



ax_dat.legend(title = r'linear/quadratic = {}'.format(round(post_odds,2)),loc='best')
fig_dat.savefig(fig_path+'fit1.pdf')

print('...done')
exit()

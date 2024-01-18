import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, tplquad, quad

#############
# Functions #
#############

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

def likelihood_gaussian_quad(a, b, c, times, datas, sigma_noise):

    num      = (datas - model_quad(times, a, b, c))**2
    gaussian = 1./np.sqrt(2.*np.pi*sigma_noise**2) * np.exp((-1./2.) * np.sum(num)/(sigma_noise)**2 )

    return gaussian

def model_lin(x,a,b):

    return a + x * b

def model_quad(x,a,b,c):

    return a + b*x +c*x*x

###############
# User inputs #
###############

show_data    = 1
zero_noise   = 0

N_data       = 100
a_inj, b_inj = 0.5, 1.1
t_min, t_max = 0.0, 1.0
sigma        = 0.2 #0.05, 0.1, 0.2, 0.6, 1

fig_path = '/home/kpapad/puk/examples/param_estiamtion/figures/MC_s02_'
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

a_min    = -2.
a_max    = 2
b_min    = -2.5
b_max    = 8.
n_points = 100000

a_vec = np.linspace(a_min, a_max, n_points)
b_vec = np.linspace(b_min, b_max, n_points)
#X,Y   = np.meshgrid(a_vec, b_vec) # grid of points

## Define the posterior as a function of the parameters only
prior_l     = (1./(a_max-a_min)) * (1./(b_max-b_min)) # we assume uniforms posteriors
posterior_l = lambda a, b: prior_l*likelihood_gaussian(a, b, time, data, sigma)

## Metropolis algorithm. The proposal distribution is uniform in a and b 
sampled_a, sampled_b = list(), list()
a0, b0 = a_inj, b_inj 
a_last, b_last = a0, b0
while len(sampled_a) != n_points:
    ai = rand.uniform(a_vec[0], a_vec[-1])
    bi = rand.uniform(b_vec[0], b_vec[-1])
    p = posterior_l(ai,bi)/posterior_l(a_last, b_last) 
    T = min(1, p)
    u = np.random.random_sample()
    if u<=T:
        sampled_a.append(ai)
        sampled_b.append(bi)
        a_last, b_last = ai, bi
    #
#
plt.figure()
counts, a_bins, b_bins, _ = plt.hist2d(sampled_a, sampled_b)

# integrate over a to get the marginalized posterior on b
# The a dimension is along the y-axis of the array 
marginalized_b = list()
marginalized_b = [ np.trapz(count, a_bins[:-1]) for count in counts.T ]

evidence_l = np.trapz(marginalized_b, b_bins[:-1])


# Find the best estimate for the parameters
a_est = np.median(sampled_a)
a_lower= a_est - np.percentile(sampled_a,5)
a_upper= np.percentile(sampled_a,95) - a_est

b_est = np.median(sampled_b)
b_lower = b_est - np.percentile(sampled_b,5)
b_upper= np.percentile(sampled_b,95) - b_est


#counts = counts / evidence 

# Having sampled a and b from their posteriors, one can consider the model as a
# random variable depending on a and b(also random variables).
# Lets plot the distribution m(a, b) = a + b*x


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
ax_l[0].axvline(a_inj, color='green', linestyle='--', label='real value')
ax_l[0].set_xlabel('a')
ax_l[0].set_ylabel(r'$P(a|H1,D,I) = \int db P(a,b|H1,D,I)$')
ax_l[0].legend()

ax_l[1].hist(sampled_b)
ax_l[1].axvline(b_est, color = "red", label='best estimate {}'.format(round(b_est,2)))
ax_l[1].axvline(b_est - b_lower , color = "blue", label = '5%')
ax_l[1].axvline(b_est + b_upper , color = "blue", label = '95%')

ax_l[1].axvline(b_inj, color='green', linestyle='--', label='real value')
ax_l[1].set_xlabel('b')
ax_l[1].set_ylabel(r'$P(b|H2,D,I) = \int da  P(a,b|H1,D,I)$')
ax_l[1].legend()

fig_l.savefig(fig_path+'linear_posterior_marginals1.pdf')
#a_est = a_vec[post_marginal_a.argmax()]
#b_est = b_vec[post_marginal_b.argmax()]



ax_dat.legend(loc='best')
#################################
# Calculate quadratic posterior #
#################################

c_min    = -6.5
c_max    = 4.5
n_points = 100000

c_vec = np.linspace(c_min, c_max, n_points)

prior_q     = (1./(a_max-a_min)) * (1./(b_max-b_min)) * (1./(c_max-c_min))
posterior_q = lambda a, b, c: prior_q*likelihood_gaussian_quad(a, b, c, time, data, sigma)

## Metropolis algorithm. The proposal distribution is uniform in a and b 
sampled_a, sampled_b, sampled_c = list(), list(), list()
a0, b0, c0 = a_inj, b_inj, 0 
a_last, b_last, c_last = a0, b0, c0
while len(sampled_a) != n_points:
    ai = rand.uniform(a_vec[0], a_vec[-1])
    bi = rand.uniform(b_vec[0], b_vec[-1])
    ci = rand.uniform(c_vec[0], c_vec[-1])
    p  = posterior_q(ai, bi, ci)/posterior_q(a_last, b_last, c_last) 
    T  = min(1, p)
    u  = np.random.random_sample()
    if u<=T:
        sampled_a.append(ai)
        sampled_b.append(bi)
        sampled_c.append(ci)
        a_last, b_last, c_last = ai, bi, ci
    #
#

counts, bins = np.histogramdd((sampled_a, sampled_b, sampled_c))
a_bins, b_bins, c_bins = bins

# each count is an array of rank two. axis=0 is the y direction and corresponds to b
# axis =1 is along the x direction and corresponds to c
# a is the z direction of the counts array 
marginal_ac = list()
marginal_ab = list()
marginal_a  = list()
for count in counts:
    int_abc_b = [ np.trapz(co, b_bins[:-1]) for co in count.T ] # integrate P(a,b,c) over b to get P(a, c)
    #int_abc_c = [ np.trapz(co, c_bins[:-1]) for co in count   ] # integrate P(a,b,c) over c to get P(a, b)
    #int_ac_c  = np.trapz(int_abc_b, c_bins[:-1])                # integrate P(a, c) over c to get P(a)

    marginal_ac.append(int_abc_b) # P(a,c)
    #marginal_ab.append(int_abc_c) # P(a,b)
    #marginal_a.append(int_ac_c)   # P(a)
#

marginal_ac = np.array(marginal_ac)
#marginal_ab = np.array(marginal_ab)

#marginal_a = np.array(marginal_a)
#marginal_b = np.trapz(marginal_ab, a_bins[:-1], axis=0)
marginal_c = np.trapz(marginal_ac, a_bins[:-1], axis=0)


evidence_q = np.trapz(marginal_c, c_bins[:-1])

# Calculate the posterior odds for the linear vs quadratic model
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
ax1[0].axvline(a_inj, color='green', linestyle='--', label='real value')
ax1[0].set_xlabel('a')
ax1[0].set_ylabel(r'$P(a|H2,D,I) = \int\int db dc P(a,b,c|H2,D,I)$')
ax1[0].legend()

ax1[1].hist(sampled_b)
ax1[1].axvline(b_est, color = "red", label='best estimate: {}'.format(round(b_est,2)))
ax1[1].axvline(b_est - b_lower , color = "blue", label = '5%')
ax1[1].axvline(b_est + b_upper , color = "blue", label = '95%')

ax1[1].axvline(b_inj, color='green', linestyle='--', label='real best estimate')
ax1[1].set_xlabel('b')
ax1[1].set_ylabel(r'$P(b|H2,D,I) = \int\int da dc P(a,b,c|H2,D,I)$')
ax1[1].legend()

ax1[2].hist(sampled_c)
ax1[2].axvline(c_est, color = "red", label='best estimate: {}'.format(round(c_est,2)))
ax1[2].axvline(c_est - c_lower , color = "blue", label = '5%')
ax1[2].axvline(c_est + c_upper , color = "blue", label = '95%')

ax1[2].axvline(0, color='green', linestyle='--', label='real best estimate')
ax1[2].set_xlabel('c')
ax1[2].set_ylabel(r'$P(c|H2,D,I) = \int\int da db P(a,b|H2,D,I)$')
ax1[2].legend()
fig1.savefig(fig_path+'quad_posteriors_marginalized1.pdf')



ax_dat.legend(title = r'linear/quadratic = {}'.format(round(post_odds,2)),loc='best')
fig_dat.savefig(fig_path+'fit1.pdf')

exit()

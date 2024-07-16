from numpy import random as rand
from matplotlib import pyplot as plt
from scipy.integrate import quad
import numpy as np
import multiprocessing as mp
from itertools import repeat
from numba import jit, prange


pi = np.pi

@jit(nopython=True)
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

## SNR Sampling functions
@jit(nopython=True)
def SNR(x):
    '''
    the Signal to Noise ratio
    probability disribution
    '''
    return 1/x**4

@jit(nopython=True)
def inverse_cdf_SNR(x):
    return -(1/(3*x))**(1/3)

@jit(nopython=True)
def sample_snr(n_data):
    sampled_SNR = []
    while len(sampled_SNR) != n_data:
        u = rand.uniform(0,1)
        snr = abs( inverse_cdf_SNR(u))
        if snr>=12:
            sampled_SNR.append(snr)
        #
    return np.array(sampled_SNR)


@jit(nopython=True, parallel=True)
def posterior(mu_j, sigma_j, measured_param, measurement_error, n_data, prior_mu, prior_s):
    '''
    Evaluate the posterior at a given point in the mu-sigma hyperspace
    mu_j (float)                   : hyper mu
    sigma_j (float)                : hyper sigma
    measured_param (iterable)      : list of (simulated) measurements for a given parameter
    measurement_error (iterable)   : list of (simulated) uncertainties for the measurements of a given parameter
    n_data (int)                   : number of observations for the parameter under study
    prior_mu (float)               : the prior on the hyper parameter mu
    prior_s (float)                : the prior on the hyper parameter sigma
    '''

    integrals = np.zeros(n_data)
    for i in prange(n_data):
        a = measured_param[i] - 4 * measurement_error[i]
        b = measured_param[i] + 4 * measurement_error[i]
        n = 1000  # number of intervals, must be even
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = gaussian(x, measured_param[i], measurement_error[i])* gaussian(x, mu_j, sigma_j)

        # Simpson's 3/8 rule
        integral = y[0] + y[n]
        for j in range(1, n, 3):
            integral += 3 * y[j] + 3 * y[j+1]
        for j in range(3, n-2, 3):
            integral += 2 * y[j]
        integral *= 3 * h / 8

        integrals[i] = integral
    result = np.prod(integrals)
    return result



def sample_posterior(n_points, mu_init, s_init, mu_jump, s_jump, delta_phi, measurement_error, prior_mu, prior_s, num_data):
    '''
    Generate MCMC samples from the posterior 
    n_points          : the number of points to sample
    mu_init           : initial mu
    s_init            : initial sigma
    mu_jump           : the sigma of the proposal distribution for mu
    sigma_jump        : the sigma of the proposal distribution for sigma
    delta_phi         : list of the simulated measurements for delta phis
    measurement_error : list of the simulated uncertainties for the corresponding delta phi
    prior_mu          : the prior on the mu
    prior_s           : the prior on the sigma
    '''
    
    hyper_mus, hyper_sigmas = list(), list()
    n_samples= 1
    mu_acc, s_acc = mu_init, s_init
    for i in range(n_points):
        mu_new= rand.normal(mu_acc, mu_jump)
        s_new= rand.normal(s_acc, s_jump)# do not hard code
        
        p_new = posterior(mu_new,s_new, delta_phi,
                measurement_error, num_data, prior_mu, prior_s)
        p_acc = posterior(mu_acc, s_acc, delta_phi,
                          measurement_error, num_data, prior_mu, prior_s) 

        p = p_new/p_acc
        T = min(1, p)
        u = rand.random_sample()
        if u <= T:
            mu_acc, s_acc = mu_new, s_new
            n_samples += 1
        #
        hyper_mus.append(mu_acc)
        hyper_sigmas.append(s_acc)
        
        print('Acceptance: {perc} ({samples_N}/{tot_N})'\
              .format(perc=(n_samples/(i+1)), samples_N=n_samples, tot_N=i+1))
        
    return (hyper_mus, hyper_sigmas) 



## Plotting
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

## Issues for discussion :
# 2. the proportionality constant of measurement error
# 3. region of integration
from numpy import random as rand
from matplotlib import pyplot as plt
from scipy.integrate import quad
import numpy as np
pi = np.pi

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

def SNR(x):
    '''
    the Signal to Noise ratio
    probability disribution
    '''
    return 1/x**4

def inverse_cdf_SNR(x):
    return -(1/(3*x))**(1/3)

def gaussian(p, mu, sigma):
    expo = -0.5*( (p - mu)/sigma )**2
    norm = 1/np.sqrt(2*pi*sigma**2)
    return norm*np.exp(expo)

init_plotting()

n_data = 100
def sample_snr(n_data):
    sampled_SNR = []
    while len(sampled_SNR) != n_data:
        u = rand.uniform(0,1)
        snr = abs( inverse_cdf_SNR(u))
        if snr>=12:
            sampled_SNR.append(snr)
        #
    return sampled_SNR
    

sampled_SNR = np.array( [  sample_snr(n_data) for times in range(3) ] ) # Generate SNRs for the 3 datasets

plt.hist(sampled_SNR[0])
plt.hist(sampled_SNR[1])
plt.hist(sampled_SNR[2])


# SNR proportionality constants 
prop_0 = np.array( [24*0.06 for num in range(n_data)] )
prop_1 = np.array( [24*0.3  for num in range(n_data)] )
prop_2 = np.array( [24*0.2  for num in range(n_data)] )

measurement_error_0 = (1/sampled_SNR[0])*prop_0
measurement_error_1 = (1/sampled_SNR[1])*prop_1
measurement_error_2 = (1/sampled_SNR[2])*prop_2
plt.figure()
plt.title('sigmas')
plt.hist(measurement_error_0)
plt.hist(measurement_error_1)
plt.hist(measurement_error_2)

deltaPhi_0 = np.array(
    [rand.normal(0, abs(measurement_error_0[i])) for i in range(n_data)]
)

deltaPhi_1 = np.array(
    [rand.normal(0., abs(m_e)) for m_e in measurement_error_1]
)
deltaPhi_2 = np.array(
    [rand.normal(0., abs(m_e)) for m_e in measurement_error_2]
)

plt.figure()
plt.title('mus')
plt.hist(deltaPhi_0)
plt.hist(deltaPhi_1)
plt.hist(deltaPhi_2)

plt.show()

def integral(mu_j, sigma_j, measured_param, measurement_error, n_data, prior_mu, prior_s):
    integrals = [
        quad( lambda p_i: gaussian(p_i, measured_param[i], measurement_error[i])*gaussian(p_i, mu_j, sigma_j),
              measured_param[i]-3*measurement_error[i], measured_param[i]+3*measurement_error[i])[0]
        for i in range(n_data) # the limit of integration is 3sigma around each p_i
    ]
    result = 1
    for integral in integrals:
        result *= integral
        
    return prior_mu*prior_s*result

prior_mu_0 = 1/0.1 # the hyper prior on the hyper mu as taken from the graph
prior_s_0 = 1/0.09 # the hyper prior on the hyper sigma as taken from the graph

prior_mu_1 = 1/0.1 # the hyper prior on the hyper mu as taken from the graph
prior_s_1 = 1/0.09 # the hyper prior on the hyper sigma as taken from the graph

prior_mu_2 = 1/0.1 # the hyper prior on the hyper mu as taken from the graph
prior_s_2 = 1/0.09 # the hyper prior on the hyper sigma as taken from the graph


def sample_posterior(n_points, mu_init, s_init, delta_phi, measurement_error, prior_mu, prior_s, num_data=n_data):
    '''
    Generate MCMC samples from the posterior 
    n_points: the number of points to sample
    mu_init: initial mu
    s_init: initial sigma
    delta_phi: list of the simulated measurements for delta phis
    measurement_error: list of the simulated uncertainties for the corresponding delta phi
    prior_mu: the prior on the mu
    prior_s: the prior on the sigma
    '''
    hyper_mus, hyper_sigmas = list(), list()
    n_samples= 1
    mu_acc, s_acc = mu_init, s_init
    for i in range(1000):
        mu_new= rand.normal(mu_acc, 1)
        s_new= rand.normal(s_acc, 5)
        p = integral(mu_new,s_new, delta_phi, measurement_error, num_data, prior_mu, prior_s)\
            /integral(mu_acc, s_acc, delta_phi, measurement_error, num_data, prior_mu, prior_s) 
        #
        T = min(1, p)
        u = rand.random_sample()
        if u <= T:
            mu_acc, s_acc = mu_new, s_new
            n_samples += 1
        #
        hyper_mus.append(mu_acc)
        hyper_sigmas.append(s_acc)
        print('Acceptance: {perc} ({samples_N}/{tot_N})'.format(perc=(n_samples/(i+1)), samples_N=n_samples, tot_N=i+1))
    return (hyper_mus, hyper_sigmas) 
    #
#
n_points = 1000
mu0_init, sigma0_init = 0, 1

hyper_mus_0, hyper_sigmas_0 = sample_posterior(n_points, mu0_init, sigma0_init, deltaPhi_0, measurement_error_0, prior_mu_0, prior_s_0)
#hyper_mus_1, hyper_sigmas_1 = sample_posterior(n_points, mu1_init, sigma1_init, deltaPhi_1, measurement_error_1, prior_mu_1, prior_s_1)
#hyper_mus_0, hyper_sigmas_2 = sample_posterior(n_points, mu2_init, sigma2_init, deltaPhi_2, measurement_error_2, prior_mu_2, prior_s_2)




plt.hist(hyper_mus_0,50, alpha=0.5, density=True, label=r'$\delta\hat\phi_0$')
#plt.hist(hyper_mus_1,50, alpha=0.5, density=True, label=r'$\delta\hat\phi_1$')
#plt.hist(hyper_mus_2,50, alpha=0.5, density=True, label=r'$\delta\hat\phi_2$')
plt.xlim(-0.3, 0.3)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\int d\sigma P(\mu, \sigma|D)$')
plt.legend(loc='best')

plt.figure()
plt.hist(hyper_sigmas_0, alpha=0.5, density=True, label=r'$\delta\hat\phi_0$')
#plt.hist(hyper_sigmas_1, alpha=0.5, density=True, label=r'$\delta\hat\phi_1$')
#plt.hist(hyper_sigmas_2, alpha=0.5, density=True, label=r'$\delta\hat\phi_2$')
#plt.xlim(0., 0.8)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\int d\mu P(\mu, \sigma|D)$')
plt.legend(loc='best')
plt.show()

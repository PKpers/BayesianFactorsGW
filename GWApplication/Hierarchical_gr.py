from numpy import random as rand
from matplotlib import pyplot as plt
from scipy.integrate import quad
import numpy as np
import multiprocessing as mp
from itertools import repeat

from hgrlib import *

pi = np.pi

init_plotting()

#####################Data simulation inputs####################################################
n_data                          = 100               # Number of data to simulate
plot_snr, plot_sigmas, plot_mus = False, False, False # Plot snr, measurement error, delta phis   
###############################################################################################


#####################Posterior sampling inputs###############################################
n_points              = 1000  # number of posterior samples
mu0_init, sigma0_init = 0, 1
# Initial hyper mu and hyper sigma for the phi0 hyper parameter 
mu0_jump, sigma0_jump = 1, 0.2
#############################################################################################

populations = 200

@jit(nopython=True)
def generate_dataset(n_data, var_idx):
    '''
    n_data (int) : The number of measurements to generate
    var_idx (int): The index number for the parameter generating measurements
    '''

    # Sample the snr
    sampled_SNR = sample_snr(n_data)

    # SNR proportionality constants
    if var_idx==0:
        prop = np.array( [24*0.06 for num in range(n_data)] )

    elif var_idx==1:
        prop = np.array( [24*0.3 for num in range(n_data)] )
    
    elif var_idx==2:
        prop = np.array( [24*0.2  for num in range(n_data)] )

    # calculate the measurement errors of the parameter
    measurement_error = (1/sampled_SNR)*prop / 2

    #calculate the paramter
    deltaPhi = np.array(
        [rand.normal(0., abs(m_e)) for m_e in measurement_error]
    )   

    return deltaPhi, measurement_error

#deltaPhi_0, measurement_error_0 = generate_dataset(n_data, 0)
#deltaPhi_1, measurement_error_1 = generate_dataset(n_data, 1)
#deltaPhi_2, measurement_error_2 = generate_dataset(n_data, 2)

## Make plots if want to
if plot_snr:
    plt.title('Sampled SNR')
    #plt.plot(x_range, 1/(x_range**4 - 12**4), label=r'$\propto\frac{1}{x^4}$')
    plt.hist(sampled_SNR[0],alpha=0.5, label='SNR0')
    plt.hist(sampled_SNR[1],alpha=0.5, label='SNR1')
    plt.hist(sampled_SNR[2],alpha=0.5, label='SNR2')
    plt.xlabel('SNR')
    plt.legend(loc='best')

if plot_sigmas:
    plt.figure()
    plt.title(r'Sampled $\sigma$')
    plt.hist(measurement_error_0, alpha=0.5, label=r'$\sigma$0')
    plt.hist(measurement_error_1, alpha=0.5, label=r'$\sigma$1')
    plt.hist(measurement_error_2, alpha=0.5, label=r'$\sigma$2')
    plt.xlabel(r'$\sigma$')
    plt.legend(loc='best')

if plot_mus:
    plt.figure()
    plt.title(r'Sampled $\mu$')
    plt.hist(deltaPhi_0,alpha=0.5, label=r'$\mu$0')
    plt.hist(deltaPhi_1,alpha=0.5, label=r'$\mu$1')
    plt.hist(deltaPhi_2,alpha=0.5, label=r'$\mu$2')
    plt.xlabel(r'$\mu$')

if any([plot_mus, plot_sigmas, plot_snr]):
    plt.legend(loc='best')
    plt.show()

    #make a new frame for other plots
    plt.figure()

#
## Sample the posterior
prior_mu_0 = 1/0.5 # the hyper prior on the hyper mu as taken from the graph
prior_s_0 = 1/0.2 # the hyper prior on the hyper sigma as taken from the graph

prior_mu_1 = 1/0.5 # the hyper prior on the hyper mu as taken from the graph
prior_s_1 = 1/0.2 # the hyper prior on the hyper sigma as taken from the graph

prior_mu_2 = 1/0.5 # the hyper prior on the hyper mu as taken from the graph
prior_s_2 = 1/0.2 # the hyper prior on the hyper sigma as taken from the graph


bests_hmu_0, bests_hsig_0, lows_hmu_0, lows_hsig_0, highs_hmu_0, highs_hsig_0 = [list() for i in range(6)]
sample_size = np.arange(10, n_data, 2)

for num in sample_size:
    best_hmu_0_dump, best_hsig_0_dump, lower_hmu_0_dump, lower_hsig_0_dump,  higher_hmu_0_dump, higher_hsig_0_dump = [list() for i in range(6)]
    for population in range(populations):
        
        deltaPhi_0, measurement_error_0 = generate_dataset(num, 0)
    
        hyper_mus_0,  hyper_sigmas_0 = sample_posterior(n_points, mu0_init, sigma0_init, mu0_jump, sigma0_jump, deltaPhi_0, measurement_error_0, prior_mu_0, prior_s_0, num)
        best_hmu_0,   best_hsig_0    = np.median(hyper_mus_0), np.median(hyper_sigmas_0)
        lower_hmu_0,  lower_hsig_0   = np.percentile(hyper_mus_0, 5), np.percentile(hyper_sigmas_0, 5)
        higher_hmu_0, higher_hsig_0  = np.percentile(hyper_mus_0, 95), np.percentile(hyper_sigmas_0, 95)

        best_hmu_0_dump.append(best_hmu_0)
        best_hsig_0_dump.append(best_hsig_0)

        lower_hmu_0_dump.append(lower_hmu_0)
        lower_hsig_0_dump.append(lower_hsig_0)

        higher_hmu_0_dump.append(higher_hmu_0)
        higher_hsig_0_dump.append(higher_hsig_0)


    bests_hmu_0.append(np.mean(best_hmu_0_dump ))
    bests_hsig_0.append(np.mean( best_hsig_0_dump ))
    lows_hmu_0.append(np.mean(lower_hmu_0_dump))
    lows_hsig_0.append(np.mean(lower_hsig_0_dump))
    highs_hmu_0.append(np.mean(higher_hmu_0_dump))
    highs_hsig_0.append(np.mean(higher_hsig_0_dump))

#

# export the results to a csv file for plotting
import csv
rows = zip(bests_hmu_0, bests_hsig_0, lows_hmu_0, lows_hsig_0, highs_hmu_0, highs_hsig_0)

with open('output.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(['bests_hmu_0', 'bests_hsig_0', 'lows_hmu_0', 'lows_hsig_0', 'highs_hmu_0', 'highs_hsig_0'])
    
    # Write the data rows
    csvwriter.writerows(rows)

exit()
#plt.plot(sample_size, bests_hmu_0, 'g--')
#plt.fill_between(sample_size, lows_hmu_0, highs_hmu_0, alpha=0.5)
#plt.show()

#hyper_mus_1, hyper_sigmas_1 = sample_posterior(n_points, mu1_init, sigma1_init, deltaPhi_1, measurement_error_1, prior_mu_1, prior_s_1, n_data)
#hyper_mus_0, hyper_sigmas_2 = sample_posterior(n_points, mu2_init, sigma2_init, deltaPhi_2, measurement_error_2, prior_mu_2, prior_s_2, n_data)

plt.hist(hyper_mus_0,20, alpha=0.5, density=True, label=r'$\delta\hat\phi_0$')
#plt.hist(hyper_mus_1,50, alpha=0.5, density=True, label=r'$\delta\hat\phi_1$')
#plt.hist(hyper_mus_2,50, alpha=0.5, density=True, label=r'$\delta\hat\phi_2$')
#plt.xlim(-0.3, 0.3)
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

## This script will be used to generate the different
## Data sets for the application of Bayesian Factors
## on combining GW data from different sources
import numpy as np
from numpy import random 
from matplotlib import pyplot as plt 
from scipy import integrate, stats
from math import isnan, inf
pi = np.pi


def gaussian_likelihood(mu, sigma, data):
    norm = 1/(np.sqrt(2*pi*sigma**2))
    expo = np.exp( -0.5*( (data - mu)/sigma)**2 )
    return norm*expo

def multiply_self(list_):
    '''
    multiplies the elements
    of the list with each other
    eg: list = [1,2,3]
    miltiply_self(list) = 1*2*3
    '''
    m = 1
    for l in list_:
        m *= l
    return m


def mc_sample_1d(l_init, l_range, num_points, f):
    '''
    samples points in 1d from a given distribution
    using metropolis hastings algorithm

    Parameters:
    l_init    : float,      The starting point of sampling
    l_range   : array like, The parameter space of sampling
    num_points: integer,    The number of points to sample
    f         : function,   The function to sample from

    Outputs:
    sampled   : list,       The list of sampled points

    '''
    
    sampled = []
    l_last = l_init
    while len(sampled) != num_points:
        li = random.uniform(l_range[0], l_range[-1])
        p = f(li)/f(l_last)
        T = min(1, p)
        u = random.random_sample()
        if u<=T:
            sampled.append(li)
            l_last = li
        #
    #
    return np.array(sampled)
    
def same_lambdas(data, sigma, param_space):
    '''
    Calculate the Bayes Factor
    assuming all the datasets
    have the same parameter
    lambda.
    Assuming Gaussian likelihood
    same sigma for all data points
    and flat prior

    Inputs:
    data       : array-like, The datasets to be analyzed
    sigma      : float,      The noise of each data set
    param_space: array_like, The bounds(param_space[0], param_space[-1]) of the parameter space  

    Outputs:
    BF         : float,      The BF 
    
    '''
    l_min, l_max = param_space[0], param_space[-1]

    prior = 1/(l_max - l_min)
    likelihood = lambda l : gaussian_likelihood(l, sigma, data) # a list containing the likelihoods for each data point
    likelihood_same = lambda l : multiply_self(likelihood(l))   # The the product of likelihoods for each data point 

    integral = integrate.quad(likelihood_same, l_min, l_max)
    #print(likelihood_same(0) / integral[0])
    #exit()
    bf = (1/prior) * likelihood_same(0) / integral[0]
    return bf

def find_percentiles(mean, std_dev):
    # Create a Gaussian distribution
    dist = stats.norm(loc=mean, scale=std_dev)

    # Find the percentiles corresponding to 0.5% and 99.5%
    lower_percentile = dist.ppf(0.005)
    upper_percentile = dist.ppf(0.995)

    return lower_percentile, upper_percentile



def uncor_lambdas(data, sigma, param_space):
    '''
    Calculate the Bayes Factor
    assuming all the datasets
    have uncorrelated parameters.

    Inputs:
    data       : array-like, The datasets to be analyzed
    sigma      : float,      The noise of each data set
    param_space: array_like, The bounds(param_space[0], param_space[-1]) of the parameter space  

    Outputs:
    BF         : float,      The BF 
    
    '''
    l_min, l_max = param_space[0], param_space[-1]
    prior = 1/(l_max - l_min)
    likelihood_ind_sampled = list()
    bf = 1
    for d in data:
        lklhd = lambda l: gaussian_likelihood(l, sigma, d)
        l_min, l_max = find_percentiles(d, sigma)
        prior = 1/(l_max - l_min)
        integral = integrate.quad(lklhd, l_min, l_max)
        bf_ = (1/prior) * lklhd(0)/integral[0]
        bf *= bf_
    #
    return bf

def cleanup(list_):
    '''
    removes inf and nan values from a list.
    '''
    clean = [ i for i in list_ if not np.isinf(i) and not np.isnan(i) ]
    return clean



n_datasets = np.arange(10, 250, 2) #number of data sets to generate
mu, sigma = 0, 1 #
l_min, l_max, l_size = -10, 10, 10000
l_vec = np.linspace(l_min, l_max, l_size)
num_sim = 10 # the number of times to simulate a given dataset

BF_GR_same, BF_NGRS_same, BF_NGRD1_same, BF_NGRD2_same  = list(), list(), list(), list()
BF_GR_diff, BF_NGRS_diff, BF_NGRD1_diff, BF_NGRD2_diff  = list(), list(), list(), list()

for num_d in n_datasets:
    GR = [ 
        np.zeros(num_d) + random.normal(mu, sigma, num_d) # GR is correct + noise
        for _ in range(num_sim)
    ]

    NoGR_same  = [
        np.ones(num_d)*0.1 + random.normal(mu, sigma, num_d) # GR is wrong. Same BGR param + noise                 
        for _ in range(num_sim)
    ]

    NoGR_diff1 = [
        random.uniform(-1, 1, num_d) + random.normal(mu, sigma, num_d)  # GR is wrong. Different BGR params + noise 
        for _ in range(num_sim)
    ]

    NoGR_diff2 = [
        random.uniform(-4, 4, num_d) + random.normal(mu, sigma, num_d)  # GR is wrong. Different BGR params + noise 
        for _ in range(num_sim)
    ]

    # Assume same parameters for each dataset
    bf_GR_same     = np.mean( [ same_lambdas(gr, sigma, (l_min, l_max)) for gr in GR] )  
    bf_NGRS_same   = np.mean( [ same_lambdas(ngrs, sigma, (l_min, l_max)) for ngrs in NoGR_same] )  
    bf_NGRD1_same  = np.mean( [ same_lambdas(ngrd,  sigma, (l_min, l_max)) for ngrd in NoGR_diff1] )
    bf_NGRD2_same  = np.mean( [ same_lambdas(ngrd,  sigma, (l_min, l_max)) for ngrd in NoGR_diff2] )

    BF_GR_same.append(bf_GR_same)
    BF_NGRS_same.append(bf_NGRS_same)
    BF_NGRD1_same.append(bf_NGRD1_same)
    BF_NGRD2_same.append(bf_NGRD2_same)

    # Assume different parameters for each data set 
    bf_GR_diff    = np.mean( [ uncor_lambdas(gr, sigma, (l_min, l_max)) for gr in GR ] ) 
    bf_NGRS_diff  = np.mean( [ uncor_lambdas(ngrs, sigma, (l_min, l_max)) for ngrs in NoGR_same ] )
    bf_NGRD1_diff = np.mean( [ uncor_lambdas(ngrd, sigma, (l_min, l_max)) for ngrd in NoGR_diff1 ] )
    bf_NGRD2_diff = np.mean( [ uncor_lambdas(ngrd, sigma, (l_min, l_max)) for ngrd in NoGR_diff2 ] )

    # Clean the resulting lists from nans of infs
    
    BF_GR_diff.append(bf_GR_diff)
    BF_NGRS_diff.append(bf_NGRS_diff)
    BF_NGRD1_diff.append(bf_NGRD1_diff)
    BF_NGRD2_diff.append(bf_NGRD2_diff)

    
#
BF_GR_diff    = cleanup(BF_GR_diff)
BF_NGRS_diff  = cleanup(BF_NGRS_diff)
BF_NGRD1_diff = cleanup(BF_NGRD1_diff)
BF_NGRD2_diff = cleanup(BF_NGRD2_diff)

N = np.linspace(0, n_datasets[-1], 1000)
plt.figure()
plt.plot(N, (l_max-l_min)*np.sqrt(N/(2*pi)),
         color='black', linestyle='--', label=r'~$\sqrt{N}$ (GR analytic)'
)

plt.plot(N, (l_max-l_min)*np.sqrt(N/(2*pi))*np.exp(-N/2),
         color='black', linestyle='-.', label=r'~$\sqrt{N} e^{-N/2}$ (Non-GR analytic)'
)

plt.scatter(n_datasets, BF_GR_same,   color='green', s = 10, label = 'GR + noise')
plt.scatter(n_datasets, BF_NGRS_same, color='purple', s = 10, label = r'Non-GR + noise, $\mu=1$ ')
plt.scatter(n_datasets, BF_NGRD1_same, color='pink', s = 10,  label = r'Non-GR + noise, $\mu\in[-1,1]$ ')
plt.xlabel('Number of datasets')
plt.ylabel('Bayes Factor')
plt.ylim(pow(10,-3), pow(10, 2)+100)
plt.xlim(1, 250)
plt.yscale('log')
plt.legend()



plt.figure()
plt.plot(N,  np.power( ( 4*np.sqrt(1/(2*pi)) ), N ),
         color='black', linestyle='--', label=r'~$x^N$ (GR analytic)'
)

plt.plot(N,  np.power( ( 0.5*np.sqrt(1/(2*pi)) ), N )*np.exp(-N/2),
         color='black',linestyle='-.', label=r'~$x^N e^{-N/2}$ (Non-GR analytic)'
)

plt.scatter(n_datasets, BF_GR_diff,   color='green', s = 10,   label = 'GR + noise')
plt.scatter(n_datasets, BF_NGRS_diff, color='pink', s = 10, label = r'Non-GR + noise, $\mu=0.1$ ')
plt.scatter(n_datasets, BF_NGRD2_diff, color='purple',   s = 10, label = r'Non-GR + noise, $\mu\in[-4,4]$ ')
plt.xlabel('Number of datasets')
plt.ylabel('Bayes Factor')
plt.yscale('log')
plt.ylim(1e-100, 1e80)
plt.xlim(0, 250)
plt.legend()

plt.show()

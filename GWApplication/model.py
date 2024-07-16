import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from numba import jit, prange
from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime
from hgrlib import sample_snr
from numpy import random as rand


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))



def my_likelihood(mu_j, sigma_j, measured_param, measurement_error):
    '''
    Evaluate the posterior at a given point in the mu-sigma hyperspace
    mu_j (float)                   : hyper mu
    sigma_j (float)                : hyper sigma
    measured_param (iterable)      : list of (simulated) measurements for a given parameter
    measurement_error (iterable)   : list of (simulated) uncertainties for the measurements of a given parameter
    '''
    
    n_data = measured_param.size
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
    #result = np.prod(integrals)
    result = integrals
    return result


def log_like(mu_j, sigma_j, measured_param, measurement_error):
    '''
    returns the log likelihood
    '''
    return np.log(my_likelihood(mu_j, sigma_j, measured_param, measurement_error))


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


# define a pytensor Op for our likelihood function


class LogLike(Op):
    def make_node(self, mu_j, sigma_j, measured_param, measurement_error) -> Apply:
        # Convert inputs to tensor variables
        mu_j = pt.as_tensor(mu_j)
        sigma_j = pt.as_tensor(sigma_j)
        measured_param = pt.as_tensor(measured_param)
        measurement_error = pt.as_tensor(measurement_error)

        #data = pt.as_tensor([measured_param, measurement_error])

        inputs = [mu_j, sigma_j, measured_param, measurement_error]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [measured_param.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        mu_j, sigma_j, measured_param, measurement_error = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = log_like(mu_j, sigma_j, measured_param, measurement_error)
        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)
        
##

def custom_dist_loglike(data, mu_j, sigma_j):

    # data, or observed is always passed as the first input of CustomDist.
    # for this case data need to be a 2 dimensional array
    # data[0] is measured_param
    # data[1] is measurement_error

    #measured_param, measurement_error = data
    return loglike_op(mu_j, sigma_j, data[0], data[1])


deltaPhi_0, measurement_error_0 = generate_dataset(100, 0)

loglike_op = LogLike()
test_out = loglike_op(0, 1, deltaPhi_0, measurement_error_0)

#pytensor.dprint(test_out, print_type=True)
#test_out.eval()
#print(log_like(0.8, 0.1, deltaPhi_0, measurement_error_0))

# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    mu_j = pm.Uniform("mu_j", lower=-1.0, upper=1.0, initval=0.8)
    sigma_j = pm.Uniform("sigma_j", lower=-1.0, upper=1.0, initval=0.1)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", mu_j, sigma_j, observed=np.array([deltaPhi_0, measurement_error_0]), logp=custom_dist_loglike
    )
    
    #ip = no_grad_model.initial_point()
    #print(no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip))
    #print(ip)
    idata_no_grad = pm.sample(1000, tune=1000)
##
# plot the traces
az.plot_trace(idata_no_grad, lines=[("mu_j", {}, 0), ("sigma_j", {}, 0)]);
    

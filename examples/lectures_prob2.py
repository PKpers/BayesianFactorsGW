import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.integrate import dblquad, tplquad
from scipy.optimize import minimize 

sig = 0.5 # we assume the error in our measurements 

## define the models and the likelihoods we calculated
def linear(x, a, b):
    return a*x+b
#

def quadratic(x, a, b, c):
    return a*x*x + b*x + c
#

def likelihood_linear(y, a, b, x, sig_ = sig):
    exponent = np.power( (y - linear(x, a, b)), 2)
    return np.exp(-0.5*np.sum(exponent)/np.power(sig_,2))
#

def likelihood_quad(y, a, b, c, x, sig_=sig):
    exponent = np.power((y - quadratic(x, a, b, c)),2)
    return np.exp(-0.5*np.sum(exponent)/np.power(sig_,2))
#

## create the data set
y = np.array([
    0.62165013, 1.02361924, 1.51161683, 0.78429014, 0.76852557,
    1.3840013, 2.98271159, 2.62484856, 2.51702153
])
x = np.linspace(0,1,9)

## To perform the integration, define first the limits of the parameter a, b and c
amin = -10.0
amax = 10.0
bmin = -10.0
bmax = 10.0
cmin = -10.0
cmax = 10.0

# evaluate the likelihoods on the measured x and y, but let them
# still remain callable on the parameters a, b and c
ll = lambda a, b: likelihood_linear(y, a, b, x)  
lq = lambda a, b, c: likelihood_quad(y, a, b, c, x)

marginal_lin = dblquad(ll, amin, amax, bmin, bmax)
marginal_quad = tplquad(lq, amin, amax, bmin, bmax, cmin, cmax)

print('the prior odds are: ', marginal_lin[0]/marginal_quad[0])

## Time to plot the posterior p(a,b|H1,D,I)
denom_lin = marginal_lin[0]

def posterior_linear_eval(a, b, x_=x, y_=y, norm=denom_lin):
    ## Evaluates the linear posterior on a grid of values a and b
    aa, bb = np.meshgrid(a, b)
    posterior_values = np.zeros_like(aa)
    for i in range(aa.shape[0]):
        for j in range(aa.shape[1]):
            posterior_values[i, j] =\
                likelihood_linear(y_, aa[i, j], bb[i, j], x_, sig) / norm
        #
    #
    return posterior_values 
    
#Create a grid of a and b values
a_values = np.linspace(amin, amax, 10)
b_values = np.linspace(bmin, bmax, 10)

# When creating a mesh grid A, B = meshgrid(a, b),
# b changes along the i direction: B[1,j], B[2,j] ...  | axis = 0
# a changes along the j direction: A[i,1], A[i 2] ...  | axis = 1
posterior_l = posterior_linear_eval(a_values, b_values)
l_marginal_a = np.trapz(posterior_l, b_values, axis=0)#integrate p(a,b) over b
l_marginal_b = np.trapz(posterior_l, a_values, axis=1)#integrate p(a,b) over a

a_est_l = a_values[l_marginal_a.argmax()]
b_est_l = b_values[l_marginal_b.argmax()]

# Create the heat map
'''
plt.figure(figsize=(15, 10))

# Plot the contour plot in the lower subplot
plt.subplot(2, 1, 2)
plt.contourf(a_values, b_values, posterior_l, cmap='viridis', levels=100)
plt.colorbar(label='P(a,b|H1,D,I)')
plt.xlabel('a')
plt.ylabel('b')

# Plot the marginal for a in the top left subplot
plt.subplot(2, 2, 1)
plt.plot(a_values, l_marginal_a, label = 'best a={}'.format(round(a_est,2)))
plt.xlabel('a')
plt.ylabel(r'$P(a|H1,D,I) = \int db P(a,b|H1,D,I)$')
plt.legend()

# Plot the marginal for b in the top right subplot
plt.subplot(2, 2, 2)
plt.plot(b_values, l_marginal_b, label = 'best b={}'.format(round(b_est,2)))
plt.xlabel('b')
plt.ylabel(r'$P(b|H1,D,I) = \int da P(a,b|H1,D,I)$')
plt.legend()

plt.tight_layout()

#plt.show()

plt.plot(x, y, 'o', label='measurements')
plt.plot(x, linear(a_est, b_est, x), 'r--',
         label = 'best estimates\n a={}, b={}'
         .format(round(a_est,2), round(b_est, 2))
)
plt.xlabel('x data')
plt.ylabel('y data')
plt.legend()
#plt.show()
'''

################################################################################
## Plot the posterior P(a,b,c|H2, D, I) for the quadratic model
denom_quad = marginal_quad[0] # the normalization constant


def posterior_quad_eval(a, b, c, x_=x, y_=y, norm=denom_quad):
    # Evaluates the quadratic posterior in a cube of the parameters a, b, c
    aaa, bbb, ccc = np.meshgrid(a, b, c)

    posterior_values = np.zeros_like(aaa)
    for i in range(aaa.shape[0]):
        for j in range(aaa.shape[1]):
            for k in range(aaa.shape[2]):
                posterior_values[i,j,k] =\
                likelihood_quad(y, aaa[i,j,k], bbb[i,j,k], ccc[i,j,k], x, sig)/norm
            #
        #
    #
    return posterior_values

a_values = np.linspace(-5, 5, 50)
b_values = np.linspace(-5, 5, 50)
c_values = np.linspace(-0.5, 2, 50)

posterior_q = posterior_quad_eval(a_values, b_values, c_values)

# When creating a 3d mesh grdid A, B, C = meshgrid(a, b c)
# C changes along the k direction C[i,j,1], C[i,j,2], ...|axis=2
# B changes along the i direction B[i,1,k], B[i,2,k], ...|axis=0
# A changes along the j direction: A[i,1,k], A[i,2,k] ...|axis=1 

q_marginal_ac = np.trapz(posterior_q, b_values, axis=0)#integrate p(a,b,c) over b
q_marginal_a = np.trapz(q_marginal_ac, c_values, axis=1)
q_marginal_c = np.trapz(q_marginal_ac, a_values, axis=0)
q_marginal_bc = np.trapz(posterior_q, a_values, axis=1)#integrate p(a,b, c) over a
q_marginal_b = np.trapz(q_marginal_bc, c_values, axis=1)

a_est = a_values[q_marginal_a.argmax()]
b_est = b_values[q_marginal_b.argmax()]
c_est = b_values[q_marginal_c.argmax()]

# plot

A_, B_, C_ = np.meshgrid(a_values, b_values, c_values)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plot = ax.scatter(A_, B_, C_,
        c=posterior_q, cmap='viridis'
)
fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10, label = 'P(a,b,c|H2,D,I)')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-0.5, 2)
plt.show()
exit()

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(a_values, q_marginal_a, label = 'best a={}'.format(round(a_est,2)))
plt.xlabel('a')
plt.ylabel(r'$P(a|H2,D,I) = \int\int db dc P(a,b,c|H2,D,I)$')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(b_values, q_marginal_b, label = 'best b={}'.format(round(b_est,2)))
plt.xlabel('b')
plt.ylabel(r'$P(b|H2,D,I) = \int\int da dc P(a,b,c|H2,D,I)$')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(b_values, q_marginal_c, label = 'best c={}'.format(round(c_est,2)))
plt.xlabel('c')
plt.ylabel(r'$P(c|H2,D,I) = \int\int da db P(a,b|H2,D,I)$')
plt.legend()

#plt.tight_layout()

plt.show()

plt.plot(x, y, 'o', label='measurements')
plt.plot(x, quadratic(x, a_est, b_est, c_est), 'r--',
         label = 'quadratic model:\n a={}, b={}, c={}'
         .format(round(a_est,2), round(b_est, 2), round(c_est, 2))
)
plt.plot(x, linear(a_est_l, b_est_l, x), 'g--',
         label = 'linear model:\n a={}, b={}'
         .format(round(a_est_l,2), round(b_est_l, 2))
)
plt.xlabel('x data')
plt.ylabel('y data')
plt.legend()
plt.show()


# Find the best paramters in the quadratic case
# to do so I will minimize the negarive log of the posterior for H2 as before
def neg_log_post_q(params, y, x, sig_):
    a, b, c = params
    return -np.log(likelihood_quad(y, a, b, c, x, sig_)/denom_quad)

# Initial guess for parameters
initial_guess = [0, 0, 0]

# Perform optimization
result = minimize(neg_log_post_q, initial_guess, args=(y, x, sig))

# Extract optimized parameters
a_opt, b_opt, c_opt = result.x

# Print the results
print("Estimated Parameters: a = {}, b = {}, c= {}".format(a_opt, b_opt, c_opt))
exit()

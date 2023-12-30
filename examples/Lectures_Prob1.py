from matplotlib import pyplot as plt
import numpy as np

def P(x_, f_, N_):
    numerator = f_*np.power(x_, N_)
    denominator = f_*np.power(x_, N_) + (1-f_)*np.power( (1-x_), N_ )
    return numerator/denominator 
#

x = 0.9
f = pow(10, -5)

num_tests = range(1, 11) # the number of tests to ... test
probs = P(x, f, num_tests)
closest_idx = np.abs(probs - 0.995).argmin()
needed_tests = num_tests[closest_idx]

plt.plot(num_tests, probs, 'o--')
plt.xlabel('Number of tests')
plt.ylabel('Probability of beeing sick')
plt.text(0.9, 0.8, 'p~0.995 after {} tests'.format(needed_tests))
plt.show()

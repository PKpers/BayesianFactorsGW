import matplotlib.pyplot as plt
import numpy as np

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

init_plotting()
x_t = np.linspace(100, 1000)

x = [100, 300, 500, 700, 1000]
y = [27.14, 37.11, 67.03, 107.7, 100.85]

plt.plot(x, y, 'b--', label=r'$\sigma=0.2$')
plt.scatter(x, y)
plt.ylabel('linear/quadratic')
plt.xlabel('Number of Points')
plt.legend(loc='best')
plt.show()

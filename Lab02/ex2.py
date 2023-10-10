import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

x1 = stats.gamma.rvs(4, scale=1 / 3,size=10000)
x2 = stats.gamma.rvs(4, scale=1 / 2,size=10000)
x3 = stats.gamma.rvs(5, scale=1 / 2,size=10000)
x4 = stats.gamma.rvs(5, scale=1 / 3,size=10000)
lat = stats.expon.rvs(scale=1 / 4,size=10000)

rez = lat + (0.25 * x1 + 0.25 * x2 + 0.3 * x3 + 0.2 * x4)

az.plot_posterior({'x1': x1, 'x2': x2,
                   'x3': x3,'x4':x4,'rez':rez})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()

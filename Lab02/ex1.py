import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
x = stats.expon.rvs(scale=1 / 4, size=10000)
y = stats.expon.rvs(scale=1 / 6, size=10000)
z = 0.4 * x + 0.6 * y  # Compunerea prin insumare a celor 2 distributii

az.plot_posterior({'x': x, 'y': y,
                   'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()

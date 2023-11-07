

import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
from scipy import stats

y=[0,5,10]
teta=[0.2, 0.5]

posteriors=[]
for i in range(0, len(y)):
    for j in range(0, len(teta)):
        with pm.Model() as model:
            n= pm.Poisson("n", mu=10)
            pm.Binomial("buyers"+str(i)+" "+str(j),n=n,p=teta[j],observed=y[i])
            idata_t = pm.sample(100, return_inferencedata=True,cores=1)
            posteriors.append((y[i], teta[j], idata_t))


fig, axes = plt.subplots(nrows=len(y), ncols=len(teta), figsize=(12, 8))

for i, (Y, theta, trace) in enumerate(posteriors):
    ax = axes[i // len(teta), i % len(teta)]
    # print(i // len(teta), i % len(teta))
    az.plot_posterior(trace,var_names=['n'], ax=ax)
    ax.set_title(f'Y={Y}, θ={theta}')

plt.tight_layout()
plt.show()

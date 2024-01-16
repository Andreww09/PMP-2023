import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

model1=az.load_arviz_data("centered_eight")
model2=az.load_arviz_data("non_centered_eight")


#ex1

print("Modelul 1:")
print(f"Numărul de lanțuri: {model1.posterior.chain.size}")
print(f"Mărimea totală a eșantionului generat: {model1.posterior.draw.size}")

az.plot_posterior(model1)
plt.suptitle("Distribuția a posteriori - Modelul 1")
plt.show()

print("Modelul 2:")
print(f"Numărul de lanțuri: {model2.posterior.chain.size}")
print(f"Mărimea totală a eșantionului generat: {model2.posterior.draw.size}")

az.plot_posterior(model2)
plt.suptitle("Distribuția a posteriori - Modelul 2")
plt.show()


#ex2
# r_hat
summaries = pd.concat([az.summary(model1, var_names=['mu','tau']),
az.summary(model2, var_names=['mu','tau'])])
summaries.index = ['centered_mu','centered_tau', 'non_centered_mu','non_centered_tau']
print(summaries)

#auto_corelatie
az.plot_autocorr(model1, var_names=['mu','tau'])
az.plot_autocorr(model2, var_names=['mu','tau'])
plt.show()


#ex3

print(f"Numarul total de divergente - Modelul 1:{model1.sample_stats.diverging.sum()}")
print(f"Numarul total de divergente - Modelul 2:{model2.sample_stats.diverging.sum()}")

_, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5),
constrained_layout=True)

for idx, tr in enumerate([model1, model2]):
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter',divergences=True, divergences_kwargs={'color':'C1'},ax=ax[idx])

ax[idx].set_title(['centered', 'non-centered'][idx])
plt.show()

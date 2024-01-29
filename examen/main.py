from types import prepare_class
from google.colab import drive
import math

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats

drive.mount('/content/drive')

# 1.a
raw_data = pd.read_csv('./Titanic.csv')
data = raw_data[['Survived', 'Pclass', 'Age']]
survived_temp = data['Survived'].values
pclass_temp = data['Pclass'].values
age_temp = data['Age'].values

survived = []
pclass = []
age = []

# se scot randurile cu valori lipsa
for i in range(0, len(pclass_temp)):
    if str(pclass_temp[i]) != "nan" and str(age_temp[i]) != "nan" and str(survived_temp[i]) != "nan":
        pclass.append(pclass_temp[i])
        age.append(age_temp[i])
        survived.append(survived_temp[i])

# se transforma in numpy arrays
survived = np.array(survived, dtype=np.float64)
pclass = np.array(pclass, dtype=np.float64)
age = np.array(age, dtype=np.float64)

# 1.b

# model liniar, se foloseste regresia logistica 
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=1)
    beta2 = pm.Normal('beta2', mu=0, sigma=1)
    mu = alpha + beta1 * pclass + beta2 * age
    teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic('bd', -alpha / beta2 - beta1 / beta2 * pclass)
    yl = pm.Bernoulli('yl', p=teta, observed=survived)
    idata = pm.sample(500, return_inferencedata=True)

az.plot_posterior(idata, var_names=['alpha', 'beta1', 'beta2'])
plt.show()

# 1.c
mean_beta1 = idata.posterior['beta1'].mean().item()
mean_beta2 = idata.posterior['beta2'].mean().item()

print(f"beta1: {mean_beta1}")
print(f"beta2: {mean_beta2}")


# beta1: -0.3712903001357159
# beta2: -0.011027284239004731

# Media lui beta1 este mai semnificativa, astfel variabila Pclass va avea o influenta mai mare asupra rezultatului


# 1.d
def compute_probability(pclass, age, hdi):
    prob = []
    alpha0 = idata.posterior['alpha'][1]
    beta1 = idata.posterior['beta1'][1]
    beta2 = idata.posterior['beta2'][1]

    for i in range(len(alpha0)):
        prob.append(1 / (1 + np.exp(-(alpha0[i] + pclass * beta1[0] + age * beta2[1]))))
    prob = np.array(prob)
    # se obtin capetele intervalului
    intervals = pm.stats.hdi(prob, hdi_prob=hdi)

    # se obtin valorile din interval
    hdi_prob = []
    for i in range(0, len(prob)):
        if prob[i] >= intervals[0] and prob[i] <= intervals[1]:
            hdi_prob.append(prob[i])

    # se afiseaza valorile ordonat
    hdi_prob = np.array(hdi_prob)
    hdi_prob = np.sort(hdi_prob)
    plt.figure()
    plt.plot(hdi_prob)
    plt.title(f'Class = {pclass} Age = {age}')
    plt.show()


compute_probability(2, 30, 0.90)


# 2.

def monte_carlo(k):
    probs = []
    for i in range(0, k):
        N = 10000
        x = stats.geom.rvs(0.3, size=N)
        y = stats.geom.rvs(0.5, size=N)
        predicted = x > y ** 2
        prob = predicted.sum() / N
        probs.append(prob)
    return probs


# 2.a 
probs = monte_carlo(1)
print(f"Predicted:{probs[0]}")
# Predicted:0.4166

# 2.b
probs = monte_carlo(30)
mean = np.mean(probs)
std = np.std(probs)
print(f"Mean: {mean}, Standard deviation: {std}")
# Mean: 0.41510333333333327, Standard deviation: 0.004502479563776191

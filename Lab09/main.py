from google.colab import drive
import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

drive.mount('/content/drive')

raw_data = pd.read_csv('./Admission.csv')
data = raw_data[["Admission", "GRE", "GPA"]]
adm = data['Admission'].values

x_n = ['GPA', 'GRE']
x_1 = np.array(data[x_n].values,dtype=np.float64)
rng = np.random.default_rng(101)


# pct a
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))
    mu = alpha + pm.math.dot(x_1, beta)
    teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])
    yl = pm.Bernoulli('yl', p=teta, observed=adm)
    idata_1 = pm.sample(100, cores=1,return_inferencedata=True,random_seed=rng)

az.plot_posterior(idata_1, var_names=['alpha','beta'])
plt.show()


# pct b
idx = np.argsort(x_1[:,0])
bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in adm])
plt.plot(x_1[:,0][idx], bd, color='k');
az.plot_hdi(x_1[:,0], idata_1.posterior['bd'], color='k',hdi_prob=0.94)
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.show()

# pct c si d
def compute_probability(gre,gpa):
  prob = []
  alpha0=idata_1.posterior['alpha'][1]
  betas = idata_1.posterior['beta'][1]

  for i in range(len(alpha0)):
      prob.append(1 / (1 + np.exp(-(alpha0[i] + gpa * betas[i][0] + gre * betas[i][1]))))
  prob = np.array(prob)
  intervals = pm.stats.hdi(prob, hdi_prob=0.95)

  hdi_prob = []
  for i in range(0,len(prob)):
      if prob[i] >= intervals[0] and prob[i] <= intervals[1]:
          hdi_prob.append(prob[i])

  hdi_prob = np.array(hdi_prob)
  hdi_prob = np.sort(hdi_prob)
  plt.figure()
  plt.plot(hdi_prob)
  plt.title(f'GRE = {gre} GPA = {gpa}')
  plt.show()


compute_probability(550,3.5)
compute_probability(500,3.2)

# in ambele cazuri studentul nu va fi admis, avand o probabilitatea prea mica de a fi admis
# primul student are o probabilitate mai mare de a fi admis dar diferenta nu este foarte semnificativa,
# acest lucru putand fi datorat inconsistentelor din setul de date



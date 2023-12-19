import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt

#ex1
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, 10]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
# print(len(mix))
az.plot_kde(np.array(mix))

#ex2
clusters = [2, 3, 4]
models = []
idatas = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means',
                          mu=np.linspace(mix.min(), mix.max(), cluster),sigma=10, shape=cluster)

        sd = pm.HalfNormal('sd', sigma=10)
        order_means = pm.Potential('order_means',pt.switch(means[1]-means[0] < 0,-np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
        idata = pm.sample(100, target_accept=0.9, random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

#ex3
for i in range(0,3):
    pm.compute_log_likelihood(idatas[i], model=models[i])

# se compara rezultatele waic pentru cele 3 modele
cmp_df = az.compare({'model_2': idatas[0], 'model_3': idatas[1],
                        'model_4': idatas[2]},
                    method='BB-pseudo-BMA', ic="waic", scale="deviance")
az.plot_compare(cmp_df)
plt.show()
# similar pentru loo
cmp_df = az.compare({'model_2': idatas[0], 'model_3': idatas[1],
                        'model_4': idatas[2]},
                    method='BB-pseudo-BMA', ic="loo", scale="deviance")
az.plot_compare(cmp_df)

# din grafice rezulta ca modelul 4 este cel mai bun
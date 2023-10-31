import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import csv

# citirea datelor
f = open("trafic.csv", "r")
data = csv.reader(f)
header = next(data)
rows = []
for row in data:
    rows.append(int(row[1]))
# print(rows)
traffic = np.array(rows)
n_traffic = len(traffic)

with pm.Model() as model:
    alpha = 1.0 / traffic.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)
    lambda_4 = pm.Exponential("lambda_4", alpha)
    lambda_5 = pm.Exponential("lambda_5", alpha)



tau1 = (7 - 4) * 60
tau2 = (8 - 4) * 60
tau3 = (16 - 4) * 60
tau4 = (19 - 4) * 60

with model:
    idx = np.arange(1200)
    lambda_ = pm.math.switch(tau1 < idx, lambda_1, pm.math.switch(tau2 < idx, lambda_2,
                                                                  pm.math.switch(tau3 < idx, lambda_3,
                                                                                 pm.math.switch(tau4 < idx, lambda_4,
                                                                                                lambda_5))))

with model:
    observation = pm.Poisson("obs", lambda_, observed=traffic)

with model:
    step = pm.Metropolis()
    trace = pm.sample(100, tune=100, step=step, return_inferencedata=False, cores=1)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
lambda_3_samples = trace['lambda_3']
lambda_4_samples = trace['lambda_4']
lambda_5_samples = trace['lambda_5']

print(lambda_1_samples.mean())
print(lambda_2_samples.mean())
print(lambda_3_samples.mean())
print(lambda_4_samples.mean())
print(lambda_5_samples.mean())


az.plot_posterior({'lambda_1': lambda_1_samples, 'lambda_2': lambda_2_samples, 'lambda_3': lambda_3_samples,
                   'lambda_4': lambda_4_samples,
                   'lambda_5': lambda_5_samples})
plt.show()

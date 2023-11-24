from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pymc as pm
import arviz as az

wins_j1 = 0  # de cate ori a castigat jucatorul 1
wins_j2 = 0
p1 = 0.5
p2 = 2 / 3.0

for i in range(0, 10000):
    stema_j1 = 0
    stema_j2 = 0
    if np.random.random() < 0.5:  # arunca primul jucator
        if np.random.random() < p1:
            stema_j1 += 1
        for j in range(0, stema_j1 + 1):  # al doilea jucator arunca de n+1 ori
            if np.random.random() < p2:
                stema_j2 += 1
    else:
        if np.random.random() < p2:  # arunca al doilea jucator
            stema_j2 += 1
        for j in range(0, stema_j2 + 1):  # primul jucator arunca de n+1 ori
            if np.random.random() < p1:
                stema_j1 += 1
    # un joc se considera castigat daca doar un jucator a obtinut stema strict de mai multe ori
    if stema_j1 > stema_j2:
        wins_j1 += 1
    elif stema_j2 > stema_j1:
        wins_j2 += 1

print(f"Probabilitatea jucator 1: {wins_j1 / (wins_j1 + wins_j2)}")
print(f"Probabilitatea jucator 2: {wins_j2 / (wins_j1 + wins_j2)}")
# Probabilitatea jucator 1: 0.34875826335536897
# Probabilitatea jucator 2: 0.651241736644631


model = BayesianNetwork([('J', 'R1'), ('R1', 'R2'),('J','R2')])
# J1 - jucatorul 1
# J2 - jucatorul 2
# S - descrie cine a inceput jocul


CPD_J = TabularCPD(variable='J', variable_card=2,
                    values=[[0.5], [0.5]])  # probabilitati egale de a alege un jucator

CPD_R1 = TabularCPD(variable='R1', variable_card=2,    # probabilitati de a obtine stema in runda 1 in functie de jucator
                   values=[[0.5, 0.33],
                           [0.5, 0.67]],
                   evidence=['J'],
                   evidence_card=[2])
CPD_R2 = TabularCPD(variable='R2', variable_card=3, # probabilitati de a obtine stema in runda 2 in functie de jucator si de prima runda
                   values=[[0.5, 0.25, 0.33, 0.1],
                           [0.5, 0.5, 0.67, 0.45],
                           [0.0, 0.25, 0.0, 0.45]],
                   evidence=['J', 'R1'],
                   evidence_card=[2, 2])


model.add_cpds(CPD_J, CPD_R1, CPD_R2)
model.get_cpds()

infer = VariableElimination(model)
posterior_p = infer.query(["J"], evidence={"R2": 1}) # stim ca s-a obtinut o singura stema in runda 2
print(posterior_p)

# +------+----------+
# | J    |   phi(J) |
# +======+==========+
# | J(0) |   0.4889 |
# +------+----------+
# | J(1) |   0.5111 |
# +------+----------+
# Al doilea jucator e mai probabil sa fi inceput jocul


data = np.random.normal(0, 10, 100) # generarea timpilor de asteptare

with pm.Model() as model1:
    mu = pm.Uniform('mu', lower=0, upper=100)
    sd = pm.HalfNormal('sd', sigma=10)
    pred=pm.Normal('pred', mu=mu, sigma=sd, observed=data)
    idata_g = pm.sample(50, return_inferencedata=True,cores=1)

az.plot_trace(idata_g, var_names=['mu','sd'])
plt.show()


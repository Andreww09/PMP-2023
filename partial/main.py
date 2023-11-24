from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az


# ex 1.1
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
        if stema_j1 >= stema_j2:
            wins_j1 += 1
        else:
            wins_j2+=1
    else:
        if np.random.random() < p2:  # arunca al doilea jucator
            stema_j2 += 1
        for j in range(0, stema_j2 + 1):  # primul jucator arunca de n+1 ori
            if np.random.random() < p1:
                stema_j1 += 1
        if stema_j2 >= stema_j1:
            wins_j2 += 1
        else:
            wins_j1 += 1

print(f"Probabilitatea jucator 1: {wins_j1 / (wins_j1 + wins_j2)}")
print(f"Probabilitatea jucator 2: {wins_j2 / (wins_j1 + wins_j2)}")
# Probabilitatea jucator 1: 0.3757
# Probabilitatea jucator 2: 0.6243


# ex 1.2
model = BayesianNetwork([('J', 'R1'), ('R1', 'R2'),('J','R2')])
# J - descrie care jucator incepe primul
# R1 - descrie rezultatele in prima runda
# R2 - descrie rezultatele in a doua runda


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

# ex 1.3
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


# ex 2.1
data = np.random.normal(0, 10, 100) # generarea timpilor de asteptare

# ex 2.2
with pm.Model() as model1:
    mu = pm.Uniform('mu', lower=0, upper=100)  # nu se pot obtine valori inafara intervalului
    sd = pm.HalfNormal('sd', sigma=10)  # sigma e o valoare suficient de mare
    pred=pm.Normal('pred', mu=mu, sigma=sd, observed=data)
    idata_g = pm.sample(50, return_inferencedata=True,cores=1)

# ex 2.3
az.plot_trace(idata_g, var_names=['mu'])    # variabila cautata este mu
plt.show()


import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
x1=[]
x2=[]
x3=[]
x4=[]

p1=0.5
p2=0.3
for j in range(0,100):
    ss=0
    bs=0
    sb=0
    bb=0
    for i in range(0,10):
        stema1=False
        if(np.random.random()<0.5):
            stema1=True

        stema2=False
        if (np.random.random() < 0.3):
            stema2 = True
        if(stema1==False and stema2==False):
            bb+=1
        if (stema1 == True and stema2 == False):
            sb += 1
        if (stema1 == False and stema2 == True):
            bs += 1
        if (stema1 == True and stema2 == True):
            ss += 1
        x1.append(bb)
        x2.append(sb)
        x3.append(bs)
        x4.append(ss)

az.plot_posterior({'bb': x1, 'sb': x2,
                   'bs': x3,'ss':x4})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()

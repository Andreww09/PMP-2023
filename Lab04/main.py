import numpy
import math
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

clients = stats.poisson.rvs(20, size=1000)
order = stats.norm.rvs(loc=2, scale=0.5, size=1000)
cookTime = stats.expon.rvs(scale=4, size=1000)

# az.plot_posterior({'clienti': clients, 'order': order,
#                    'cook': cookTime})
# plt.show()

#ex2
def served_percentage(alpha):
    count = 0
    order = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = stats.expon.rvs(scale=alpha, size=1000)
    total = order + cookTime
    for i in total:
        if i < 15:
            count += 1
    return count / len(total)


def calculate_alpha():
    alpha = 1 / 60
    while served_percentage(alpha) > 0.95:
        alpha += 1 / 60
    alpha -= 1 / 60
    return alpha

#ex3
def average_time():
    order = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = stats.expon.rvs(scale=calculate_alpha(), size=1000)
    total=order+cookTime
    return sum(total)/len(total)

print(calculate_alpha())
print(average_time())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from scipy import stats
import pymc as pm


def calculate_mean():
    order = stats.norm.rvs(loc=2, scale=0.5, size=1000)
    cookTime = stats.expon.rvs(scale=3, size=1000)
    total = order + cookTime
    return sum(total) / len(total)

# generarea esantionului
data = []
for i in range(0, 100):
    data.append(calculate_mean())

#lambda asteptat: 0.20013492648088815
print(1/(sum(data) / len(data)))

with pm.Model() as model:
    alpha = pm.Exponential("alpha",sum(data) / len(data) )

with model:
    observation = pm.Exponential("obs", alpha, observed=data)

with model:
    step = pm.Metropolis()
    trace = pm.sample(1000, tune=500, step=step, return_inferencedata=False,cores=1)

#lambda obtinut: 0.20243524335808175
print(trace[alpha].mean())


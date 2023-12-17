import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

az.style.use('arviz-darkgrid')

raw_data = np.loadtxt('./dummy.csv')
x_1 = raw_data[:, 0]
y_1 = raw_data[:, 1]

raw_data_1 = pd.read_csv('./howell.csv')
data = raw_data_1[['height', 'weight', 'age', 'male']]
x1 = data['height'].values
x2 = data['weight'].values

x_2 = np.array(x1, dtype=np.float64)
y_2 = np.array(x2, dtype=np.float64)

# obtine datele x_1s si y_1s folosite in modele in functie de fisier si de order
def get_data(data_file, order):
    if (data_file == 'dummy'):
        x, y = (x_1, y_1)
    else:
        x, y = (x_2, y_2)
    x_1p = np.vstack([x ** i for i
                      in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y - y.mean()) / y.std()
    return (x_1s, y_1s)

# ex1
order=5
sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
# model_p va lua fiecare valoare sd din lista si se va genera un grafic diferit
sds = [sd, 100, 10]
# ex2 se repeta toti pasii si pentru fisierul howell care contine 500+ puncte
for data_file in ['dummy', 'howell']:
    x_1s, y_1s = get_data(data_file, order=order)
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    # model liniar
    with pm.Model() as model_l:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10)
        teta = pm.HalfNormal('teta', 5)
        mu = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=mu, sigma=teta, observed=y_1s)
        idata_l = pm.sample(50, return_inferencedata=True)


    # modelul de order=5 va lua pe rand cele 3 valori pentru sd
    for sd_i in sds:
        with pm.Model() as model_p:
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=sd_i, shape=order)
            teta = pm.HalfNormal('teta', 5)
            mu = alpha + pm.math.dot(beta, x_1s)
            y_pred = pm.Normal('y_pred', mu=mu, sigma=teta, observed=y_1s)
            idata_p = pm.sample(50, return_inferencedata=True)

        alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
        beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
        idx = np.argsort(x_1s[0])
        y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
        plt.plot(x_1s[0][idx], y_p_post[idx], 'C1', label=f'sd {sd_i}, order {order}')
        plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
        plt.legend()
        plt.show()

    # ex3 un model patratic
    x_1s, y_1s = get_data(data_file, order=2)
    with pm.Model() as model_square:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        teta = pm.HalfNormal('teta', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=teta, observed=y_1s)
        idata_square = pm.sample(50, return_inferencedata=True)

    # un model cubic
    x_1s, y_1s = get_data(data_file, order=3)
    with pm.Model() as model_cubic:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
        teta = pm.HalfNormal('teta', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=teta, observed=y_1s)
        idata_cubic = pm.sample(50, return_inferencedata=True)


    pm.compute_log_likelihood(idata_l, model=model_l)
    pm.compute_log_likelihood(idata_square, model=model_square)
    pm.compute_log_likelihood(idata_cubic, model=model_cubic)
    # se compara rezultatele waic pentru cele 3 modele: liniar, patratic si cubic
    cmp_df = az.compare({'model_linear': idata_l, 'model_square': idata_square,
                         'model_cubic': idata_cubic},
                        method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df)
    plt.show()
    # similar pentru loo
    cmp_df = az.compare({'model_linear': idata_l, 'model_square': idata_square,
                         'model_cubic': idata_cubic},
                        method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(cmp_df)
    plt.show()

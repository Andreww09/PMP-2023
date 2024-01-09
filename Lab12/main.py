import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats


# ex1
def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    # prior = (grid<= 0.5).astype(int)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (100, 30))
points = 100
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ');
plt.show()

# ex2
mean = []
std = []
N_values = [100, 1000, 10000]

for N in N_values:
    errors = []
    for i in range(0, 10):
        x, y = np.random.uniform(-1, 1, size=(2, N))
        inside = (x ** 2 + y ** 2) <= 1
        pi = inside.sum() * 4 / N
        error = abs((pi - np.pi))
        errors.append(error)
    mean.append(np.mean(errors))
    std.append(np.std(errors))

mean = np.array(mean)
std = np.array(std)
print(mean)
print(std)
# Plot using plt.errorbar()
plt.errorbar(N_values, mean, yerr=std, fmt='o', capsize=5)
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.xlabel('N')
plt.ylabel('Mean Error (%)')
plt.title('Mean Error and Standard Deviation for Different N values')
plt.show()


# ex3
def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


beta_params = [(1, 1), (20, 20), (1, 4)]
for a, b in beta_params:
    func = stats.beta(a, b)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()

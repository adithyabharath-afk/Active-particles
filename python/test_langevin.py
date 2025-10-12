import numpy as np

def simulate_langevin_milstein(ensembles=20000, h=0.005, steps=200, seed=42):
    rng = np.random.default_rng(seed)
    c = 1.0
    r = np.sqrt(2.0)   # dv = -x - v + sqrt(2) * g(t)
    f = lambda x: -x

    x = np.zeros(ensembles)
    v = np.zeros(ensembles)
    sqrt_h = np.sqrt(h)

    for n in range(steps):
        nn = rng.standard_normal(ensembles)  # independent standard normals
        gn = rng.standard_normal(ensembles)
        # An per eq (17): 1/2 * h^2 * (f - c*v) + r * h^{3/2} * (1/2 * n + 1/(2*sqrt(3)) * g)
        An = 0.5 * h**2 * (f(x) - c * v) + r * h**1.5 * (0.5 * nn + 0.5 / np.sqrt(3.0) * gn)
        print(steps)

        x_new = x + h * v + An
        v_new = v + 0.5 * h * (f(x_new) + f(x)) - h * c * v + sqrt_h * r * nn - c * An

        x, v = x_new, v_new

    return (x**2 + v**2).mean()
print(simulate_langevin_milstein())
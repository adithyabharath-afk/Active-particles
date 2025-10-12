import numpy as np

def langevin_eq23(ensembles=20000, h=0.005, steps=200, seed=42):
    """
    Langevin integrator using equation (23) from the paper.
    Returns the ensemble average of x^2 + v^2 after time t = steps*h.
    """

    rng = np.random.default_rng(seed)

    # Parameters for the harmonic oscillator test (as in the paper, Fig. 1)
    gamma = 1.0             # friction coefficient
    sigma = np.sqrt(2.0)    # noise amplitude
    f = lambda x: -x        # force = -x

    # Initial conditions
    x = np.zeros(ensembles)
    v = np.zeros(ensembles)

    sqrt_h = np.sqrt(h)

    for n in range(steps):
        # Two independent Gaussians for each trajectory at each step
        xi = rng.standard_normal(ensembles)
        eta = rng.standard_normal(ensembles)

        # ----- Step 1: v^{n+1/2} -----
        v_half = (
            v
            + 0.5 * h * f(x)                    # 1/2 h f(x^n)
            - 0.5 * h * gamma * v               # -1/2 h γ v^n
            + 0.5 * sqrt_h * sigma * xi         # +1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (f(x) - gamma * v)   # -1/8 h^2 γ (f(x^n) - γ v^n)
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta)
        )

        # ----- Step 2: x^{n+1} -----
        x_new = (
            x
            + h * v_half                        # + h v^{n+1/2}
            + h**1.5 * sigma * (1/(2*np.sqrt(3))) * eta   # + h^{3/2} σ (1/(2√3)) η^n
        )

        # ----- Step 3: v^{n+1} -----
        v_new = (
            v_half
            + 0.5 * h * f(x_new)                # + 1/2 h f(x^{n+1})
            - 0.5 * h * gamma * v_half          # - 1/2 h γ v^{n+1/2}
            + 0.5 * sqrt_h * sigma * xi         # + 1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (f(x_new) - gamma * v_half)  # -1/8 h^2 γ (f(x^{n+1}) - γ v^{n+1/2})
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta)
        )

        # Update for next step
        x, v = x_new, v_new

    # Observable: ensemble average of x^2 + v^2
    vals = x**2 + v**2
    mean = vals.mean()
    stderr = vals.std(ddof=1) / np.sqrt(ensembles)

    return mean, stderr


# Example test: t=1 with h=0.005, 200 steps, 20000 trajectories
if __name__ == "__main__":
    mean, stderr = langevin_eq23()
    print(f"E[x^2+v^2] ≈ {mean:.6f} ± {stderr:.6f}")

import numpy as np
from molsim import Simulation
from scipy.spatial.distance import cdist

# ---------- set same params as in your integrator ----------
N = 100
rho = 0.53/(np.pi/4)
L = np.sqrt(N / rho)
r_c = (2)**(1/6)
r_c_dip = L/2 - 1
B_scale = 1e-5
mu0_over_4pi = 1e-7
epsilon = 1.0
sigma = 1.0
target_U = 1.0
T = 1.0/target_U
gamma_r = 3.0
Dr = T / gamma_r

# ------- put your old raw nu (what you passed before conversion) here -------
nu_raw_before = 1.0   # <--- change to the number you used before (if unknown, keep 1)

# ------- the converted nu you currently pass (what your integrator calculates) -------
mu_paper = 1.0        # paper value
nu_converted = mu_paper * np.sqrt(mu0_over_4pi * epsilon / (B_scale**2 * sigma**3))

print("nu_raw_before =", nu_raw_before)
print("nu_converted  =", nu_converted)

def compute_S_for_nu(nu):
    sim = Simulation(L, r_c, r_c_dip, N, float(nu), 0.0, 0.0)

    # make small-noise lattice snapshot
    def make_snapshot(N, L, small_noise=1e-2):
        n = int(round(np.sqrt(N)))
        a = L / n
        coords = []
        for i in range(n):
            for j in range(n):
                coords.append([i*a + small_noise*(np.random.rand()-0.5),
                               j*a + small_noise*(np.random.rand()-0.5)])
        coords = np.array(coords)[:N]
        orients = np.random.randn(N,2)
        orients /= np.linalg.norm(orients, axis=1)[:,None]
        return coords, orients

    coords, orients = make_snapshot(N, L)
    forces, torques = sim.force2dhp(coords.flatten().tolist(), orients.flatten().tolist())
    torques = np.array(torques)
    tau_mean = np.mean(np.abs(torques))
    omega_det = tau_mean / gamma_r
    S = omega_det / Dr   # equivalent to tau_mean / T

    # quick polar + cluster diagnostics
    P = np.linalg.norm(np.sum(orients,axis=0)) / N
    dmat = cdist(coords, coords)
    r_thresh = 1.5
    adj = (dmat < r_thresh).astype(int)
    np.fill_diagonal(adj, 0)
    visited = set(); clusters = []
    for i in range(N):
        if i in visited: continue
        stack = [i]; comp=[]
        while stack:
            u=stack.pop()
            if u in visited: continue
            visited.add(u); comp.append(u)
            neigh = np.where(adj[u])[0].tolist()
            for v in neigh:
                if v not in visited:
                    stack.append(v)
        clusters.append(comp)
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    return tau_mean, S, P, cluster_sizes[:5]

print("\nDiagnostics for converted nu:")
print(compute_S_for_nu(nu_converted))

print("\nDiagnostics for raw old nu:")
print(compute_S_for_nu(nu_raw_before))



